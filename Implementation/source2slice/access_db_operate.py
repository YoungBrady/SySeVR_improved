## -*- coding: utf-8 -*-
from __future__ import print_function
from joern.all import JoernSteps
from igraph import *
from general_op import *
import pickle
from py2neo.packages.httpstream import http
http.socket_timeout = 9999

def get_all_use_bydefnode(db, node_id):
    query_str = "g.v(%d).in('USE')" % node_id
    results = db.runGremlinQuery(query_str)
    list_re = []
    for re in results:
        if re.properties['type'] == 'Statement':
            continue
        else:
            list_re.append(re)

    return list_re


def get_all_def_bydefnode(db, node_id):
    query_str = "g.v(%d).in('DEF')" % node_id
    results = db.runGremlinQuery(query_str)
    list_re = []
    for re in results:
        if re.properties['type'] == 'Statement':
            continue
        else:
            list_re.append(re)

    return list_re


def get_exprstmt_node(db):
    query_expr_str = "queryNodeIndex('type:ExpressionStatement')"
    #results = db.runGremlinQuery(query_expr_str)
    results_1 = db.runGremlinQuery(query_expr_str)

    query_iddecl_str = 'queryNodeIndex("type:IdentifierDeclStatement")'
    results_2 = db.runGremlinQuery(query_iddecl_str)

    results = results_1 + results_2

    return results
    

def get_pointers_node(db):
    list_pointers_node = []
    query_iddecl_str = 'queryNodeIndex("type:IdentifierDeclStatement")' #查询标识符

    results = db.runGremlinQuery(query_iddecl_str)

    if results != []:
        for re in results:
            code = re.properties['code']
            if code.find(' = ') != -1:
                code = code.split(' = ')[0]

            if code.find('*') != -1: #指针匹配
                list_pointers_node.append(re)
    
    query_param_str = 'queryNodeIndex("type:Parameter")'
    results = db.runGremlinQuery(query_param_str)
    if results != []:
        for re in results:
            code = re.properties['code']
            if code.find(' = ') != -1:
                code = code.split(' = ')[0]
                
            if code.find('*') != -1:
                list_pointers_node.append(re)
    

    return list_pointers_node


def get_arrays_node(db):
    list_arrays_node = []
    query_iddecl_str = "queryNodeIndex('type:IdentifierDeclStatement')"
    results = db.runGremlinQuery(query_iddecl_str)
    if results != []:
        for re in results:
            code = re.properties['code']
            if code.find(' = ') != -1:
                code = code.split(' = ')[0]

            if code.find(' [ ') != -1:   #数组匹配
                list_arrays_node.append(re)
    
    query_param_str = "queryNodeIndex('type:Parameter')"
    results = db.runGremlinQuery(query_param_str)
    if results != []:
        for re in results:
            code = re.properties['code']
            if code.find(' = ') != -1:
                code = code.split(' = ')[0]

            if code.find(' [ ') != -1:
                list_arrays_node.append(re)
    

    return list_arrays_node


def get_def_node(db, cfg_node_id):
    query_str = "g.v(%d).out('DEF')" % cfg_node_id
    results = db.runGremlinQuery(query_str)
    return results

def get_use_node(db, cfg_node_id):
    query_str = "g.v(%d).out('USE')" % cfg_node_id
    results = db.runGremlinQuery(query_str)
    return results

def getNameByNodeid(db, cfg_node_id):
    query_str = "g.v(%d)" % cfg_node_id
    results = db.runGremlinQuery(query_str)
    return results['name']

def getFunctionNodeByName(db, funcname):
    query_str = "queryNodeIndex('type:Function AND name:%s')" % funcname
    results = db.runGremlinQuery(query_str)
    return results

def getFunctionNodeByNameInFile(db, funcname, filepath):
    filepath = '*/'+ filepath
    query_str = "queryNodeIndex('type:File AND filepath:%s').id" % filepath
    file_ids = db.runGremlinQuery(query_str)
    if file_ids == []:
        return False

    func_node = []  
    for file_id in file_ids:
        list_func_node = getFuncNodeByFile(db, file_id)
        for node in list_func_node:
            if node['name'] == funcname:
                func_node.append(node)

    return func_node


def get_parameter_by_funcid(db, func_id):
    query_str = "g.v(%d).out('IS_FUNCTION_OF_CFG').out('CONTROLS').filter{ it.type == 'Parameter' }.id" % func_id
    results = db.runGremlinQuery(query_str)
    return results


def isNodeExist(g, nodeName):
    if not g.vs:
        return False
    else:
        return nodeName in g.vs['name']


def getALLFuncNode(db):
    query_str = "queryNodeIndex('type:Function')"
    results = db.runGremlinQuery(query_str)
    return results


def getFuncNode(db, func_name):
    query_str = 'getFunctionsByName("' + func_name + '")'
    func_node = db.runGremlinQuery(query_str)
    return func_node
    

def getFuncFile(db, func_id):
    query_str = "g.v(%d).in('IS_FILE_OF').filepath" % func_id
    try:
        ret = db.runGremlinQuery(query_str)
    except Exception as e:
        return False
    return ret[0]


def getCFGNodes(db, func_id):
    query_str = 'queryNodeIndex("functionId:%s AND isCFGNode:True")' % func_id
    cfgNodes = db.runGremlinQuery(query_str)
    
    return cfgNodes

def getDDGEdges(db, func_id):
    query_str = """queryNodeIndex('functionId:%s AND isCFGNode:True').outE('REACHES')""" % (func_id)    #data dependency
    ddgEdges = db.runGremlinQuery(query_str)
    return ddgEdges


def getCDGEdges(db, func_id):
    query_str = """queryNodeIndex('functionId:%s AND isCFGNode:True').outE('CONTROLS')""" % (func_id)   #control dependency
    cdgEdges = db.runGremlinQuery(query_str)
    return cdgEdges


def getCFGEdges(db, func_id):
    query_str = """queryNodeIndex('functionId:%s AND isCFGNode:True').outE('FLOWS_TO')""" % (func_id)   #control flow
    cfgEdges = db.runGremlinQuery(query_str)
    return cfgEdges


def drawGraph(db, edges, func_entry_node, graph_type):
    g = Graph(directed=True)
    func_id = func_entry_node._id
    filepath = getFuncFile(db, func_id)

    for edge in edges:
        if edge.start_node.properties['code'] == 'ENTRY':
            startNode = str(edge.start_node.properties['functionId'])
        else:
            startNode = str(edge.start_node._id)

        if edge.start_node.properties['code'] == 'ERROR':
            continue

        if isNodeExist(g, startNode) == False:
            if edge.start_node.properties['code'] == 'ENTRY':
                node_prop = {'code': func_entry_node.properties['name'], 'type': func_entry_node.properties['type'],
                         'location': func_entry_node.properties['location'], 'filepath':filepath, 'functionId':str(edge.start_node.properties['functionId'])}
            else:
                node_prop = {'code': edge.start_node.properties['code'], 'type': edge.start_node.properties['type'],
                         'location': edge.start_node.properties['location'], 'filepath':filepath, 'functionId':str(edge.start_node.properties['functionId'])}
            g.add_vertex(startNode, **node_prop)#id is 'name'

        endNode = str(edge.end_node._id)
        if isNodeExist(g, endNode) == False:
            if graph_type == 'pdg' and edge.end_node.properties['code'] == 'EXIT':
                continue

            if edge.end_node.properties['code'] == 'ERROR':
                continue

            node_prop = {'code': edge.end_node.properties['code'], 'type': edge.end_node.properties['type'],
                         'location': edge.end_node.properties['location'], 'filepath':filepath, 'functionId':str(edge.end_node.properties['functionId'])}
            g.add_vertex(endNode, **node_prop)

        if graph_type == 'pdg':
            edge_prop = {'var': edge.properties['var']}
        else:
            edge_prop = {'var': edge.properties['flowLabel']}          
        g.add_edge(startNode, endNode, **edge_prop)

    return g


def translatePDGByNode(db, func_node):
    func_id = func_node._id
    ddgEdges = getDDGEdges(db, func_id)
    cdgEdges = getCDGEdges(db, func_id)
    Edges = ddgEdges + cdgEdges
    graph_type = 'pdg'
    g = drawGraph(db, Edges, func_node, graph_type)

    return g


def translateCFGByNode(db, func_node):
    func_id = func_node._id
    Edges = getCFGEdges(db, func_id)
    graph_type = 'cfg'
    g = drawGraph(db, Edges, func_node, graph_type)

    return g

    
def getUSENodesVar(db, func_id):
    query = "g.v(%s).out('USE').code" % func_id
    ret = db.runGremlinQuery(query)
    if ret == []:
        return False
    else:
        return ret


def getDEFNodesVar(db, func_id):
    query = "g.v(%s).out('DEF').code" % func_id
    ret = db.runGremlinQuery(query)
    if ret == []:
        return False
    else:
        return ret


def getUseDefVarByPDG(db, pdg):
    #找到PDG图里面节点中的语句里面的定义的数据和使用的数据之间的关系
    dict_cfg2use = {}
    dict_cfg2def = {}
    #print pdg
    #need_to_addedge_node = []
    for node in pdg.vs:
        if node['type'] == 'Function':
            continue
            
        func_id = node['name']
        use_node = getUSENodesVar(db, func_id)
        def_node = getDEFNodesVar(db, func_id)

        # 流操作符
        if " << " in node['code'] and "cin" not in node['code']:
            values = node['code'].split(" << ")
            for i in range(len(values)):
                values[i] = values[i].split(" [ ")[0].replace('*', '')
            use_node = []
            def_node = values[0:-1]
            use_node.append(values[-1])
        if " >> " in node['code'] and "cout" not in node['code']:
            values = node['code'].split(" >> ")
            for i in range(len(values)):
                values[i] = values[i].split(" [ ")[0].replace('*', '')
            use_node = []
            def_node = values[1:]
            use_node.append(values[0])

        if node['type'] == 'Statement':
            if def_node == False:
                code = node['code'].replace('\n', ' ')
                if code.find(" = ") != -1:
                    value = code.split(" = ")[0].strip().split(' ')
                    if value[-1] == ']':
                        newvalue = code.split(" [ ")[0].strip().split(' ')
                        if '->' in newvalue:
                            a_index = newvalue.index('->')
                            n_value = ' '.join([newvalue[a_index-1], '->', newvalue[a_index+1]])
                            newvalue[a_index-1] = n_value
                            del newvalue[a_index]
                            del newvalue[a_index]

                        def_node = newvalue

                    else:
                        if '->' in value:
                            a_index = value.index('->')
                            n_value = ' '.join([value[a_index-1], '->', value[a_index+1]])
                            ob_value = value[a_index-1]
                            value[a_index-1] = n_value
                            del value[a_index]
                            del value[a_index]
                            value.append(ob_value.replace('*', ''))

                        def_node = value

                    #need_to_addedge_node.append(node['name'])

            if use_node == False:
                if code.find(" = ") != -1:
                    value = code.split(" = ")[1].strip().split(' ')
                    newvalue = []
                    for v in value:
                        if v == '*' or v == '+' or v == '-' or v == '->' or v == '(' or v == ')' or v == '[' or v == ']' or v == '&' or v == '.' or v == '::' or v == ';' or v == ',':
                            continue
                        else:
                            newvalue.append(v.strip())

                else:
                    value = code.split(' ')
                    newvalue = []
                    for v in value:
                        if v == '*' or v == '+' or v == '-' or v == '->' or v == '(' or v == ')' or v == '[' or v == ']' or v == '&' or v == '.' or v == '::' or v == ';' or v == ',':
                            continue
                        else:
                            newvalue.append(v.strip())

                use_node = newvalue


        if use_node:
            use_node = [code.replace('*', '').replace('&', '').strip() for code in use_node]

        if def_node:
            def_node = [code.replace('*', '').replace('&', '').strip() for code in def_node]

        else:#add define node
            new_def_node = getReturnVarOfAPI(node['code'])#get modify value of api_func
            if new_def_node:
                def_node = []
                for code in new_def_node:
                    new_code = code.replace('*', '').replace('&', '').strip()
                    def_node.append(new_code)

                    if new_code not in use_node:
                        use_node.append(new_code)

        if use_node:
            dict_cfg2use[node['name']] = use_node

        if def_node:
            dict_cfg2def[node['name']] = def_node

    return dict_cfg2use, dict_cfg2def


def getFuncNodeByFile(db, filenodeID):  
    query_str = 'g.v(%d).out("IS_FILE_OF")' % filenodeID
    results = db.runGremlinQuery(query_str)
    _list = []
    for re in results:
        if re.properties['type'] == 'Function':
            _list.append(re)
        else:
            continue

    return _list

def getFuncNodeByFilepath(db, filepath):
    filepath = '*/'+ filepath
    query_str = "queryNodeIndex('type:File AND filepath:%s').id" % filepath
    file_ids = db.runGremlinQuery(query_str)
    if file_ids == []:
        return False

    list_all_func_node = []  
    for file_id in file_ids:
        list_func_node = getFuncNodeByFile(db, file_id)
        list_all_func_node += list_func_node

    return list_all_func_node


def getAllFuncfileByTestID(db, testID):
    testID = '*/'+ testID + '/*'
    query_str = "queryNodeIndex('type:File AND filepath:%s').id" % testID
    results = db.runGremlinQuery(query_str)
    return results


def get_calls_id(db, func_name):
    query_str = 'getCallsTo("%s").id' % func_name
    results = db.runGremlinQuery(query_str)
    return results


def getCFGNodeByCallee(db, node_ast_id):
    #print "start"
    query_str = "g.v(%s).in('IS_AST_PARENT')" % node_ast_id
    try:

        results = db.runGremlinQuery(query_str)
    except:
        return None
    #print "end"
    if results == []:
        return None

    for node in results:
        if 'isCFGNode' in node.properties and node.properties['isCFGNode'] == 'True':
            return node
        else:
            node = getCFGNodeByCallee(db, node._id)
    
    return node


def getCalleeNode(db, func_id):
    query_str = "queryNodeIndex('type:Callee AND functionId:%d')" % func_id
    results = db.runGremlinQuery(query_str)
    return results

def get_all_calls_node(db, testID):
    list_all_callee_node = []
    func_pointer_dict = {}
    for node in getFuncNodeInTestID(db, testID):
        func_id = node._id
        list_callee_func = getCalleeNode(db, func_id)
        if not os.path.exists(os.path.join("pdg_db", testID, node.properties['name'] + '_' + str(node._id))):
            continue
        fin = open(os.path.join("pdg_db", testID, node.properties['name'] + '_' + str(node._id)))
        pdg = pickle.load(fin)
        fin.close()
        # process function pointer
        for v in pdg.vs():
            if v['code'].find('=') != -1 and v['code'].count('=') == 1:
                # print v['code']
                left = v['code'].split('=')[0]
                right = v['code'].split('=')[1]
                # 函数指针的定义方式为：函数返回值类型 (* 指针变量名) (函数参数列表);
                # 函数指针的初始化、赋值:void (*funcp)(void) = &myfunc; void (*funcp)(void); funcp = &myfunc;
                if left.count(')') >= 2:
                    left = left.split(')')[-3]
                    left = left.split('(')[-1]
                left = left.replace('*','').replace(' ','')
                right = right.replace('&','').replace(' ','').replace(';','')
                for callee_func in list_callee_func:
                    if callee_func['code'] == left:
                        for func_node in getFuncNodeInTestID(db, testID):
                            if func_node['name'] == right:
                                func_pointer_dict[callee_func['code']] = func_node['name']
                                callee_func['code'] = func_node['name']
                                break

        list_all_callee_node += list_callee_func

    if list_all_callee_node == []:
        return [],{}
    else:
        return [(str(node._id), node.properties['code'], str(node.properties['functionId'])) for node in list_all_callee_node],func_pointer_dict

# def get_all_calls_node(db, testID):
#     list_all_funcID = [node._id for node in getFuncNodeInTestID(db, testID)]
#     print "list_all_funcID: ", list_all_funcID
#     list_all_callee_node = []
#     for func_id in list_all_funcID:#allfile in a testID
#         list_all_callee_node += getCalleeNode(db, func_id)
#         print list_all_callee_node

#     if list_all_callee_node == []:
#         return False
#     else:
#         return [(str(node._id), node.properties['code'], str(node.properties['functionId'])) for node in list_all_callee_node]


def getFuncNodeInTestID(db, testID):
    list_all_file_id = getAllFuncfileByTestID(db, testID)
    if list_all_file_id == []:
        return False

    list_all_func_node = []  

    for file_id in list_all_file_id:
        list_func_node = getFuncNodeByFile(db, file_id)
        list_all_func_node += list_func_node

    return list_all_func_node


def getClassByObjectAndFuncID(db, objectname, func_id):
    #print objectname, func_id
    all_cfg_node = getCFGNodes(db, func_id)
    for cfg_node in all_cfg_node:
        if cfg_node.properties['code'] == objectname and cfg_node.properties['type'] == 'Statement':
            # print objectname, func_id, cfg_node.properties['code'], cfg_node._id
            query_str_1 = "queryNodeIndex('type:Statement AND code:%s AND functionId:%s')" % (objectname, func_id)
            results_1 = db.runGremlinQuery(query_str_1)
            if results_1 == []:
                return False
            else:
                ob_cfgNode = results_1[0]

            location_row = ob_cfgNode.properties['location'].split(':')[0]

            query_str_2 = "queryNodeIndex('type:ExpressionStatement AND functionId:%s')" % func_id
            results_2 = db.runGremlinQuery(query_str_2)
            if results_2 == []:
                return False

            classname = False
            for node in results_2:
                # print node.properties['location'].split(':')[0], location_row
                
                if node.properties['location']!=None and node.properties['location'].split(':')[0] == location_row:
                    classname = node.properties['code']
                    break
                
                else:
                    continue

            return classname

        # elif cfg_node.properties['code'].find(' '+objectname+' = new') != -1:
        #     temp_value = cfg_node.properties['code'].split(' '+objectname+' = new')[1].replace('*', '').strip()
        elif cfg_node.properties['code'].find(objectname) != -1 and cfg_node.properties['type'] == 'IdentifierDeclStatement':
            temp_value = ''
            classname = ''
            if '=' in cfg_node.properties['code']:
                if cfg_node.properties['code'].find(' '+objectname+' = ') != -1:
                    temp_value = cfg_node.properties['code'].split(' '+objectname+' = ')[1].replace('*', '').replace('new','').strip()
            # else:
            #     temp_value = cfg_node.properties['code'].split(objectname)[0].replace('*', '').replace('&', '').strip()
            if temp_value.split(' ')[0] != 'const':
                classname = temp_value.split(' ')[0]
            else:
                classname = temp_value.split(' ')[1]
            if classname != '':
                return classname
        elif cfg_node.properties['code'].find(objectname) != -1 and cfg_node.properties['type'] == 'Parameter':
            classname = cfg_node.properties['code'].split(' ')[0].replace('*', '').replace('&','').strip()
            return classname


def getDeleteNode(db, func_id):
    query_str = "queryNodeIndex('code:delete AND functionId:%d')" % func_id
    results = db.runGremlinQuery(query_str)
    return results


def get_all_delete_node(db, testID):
    list_all_funcID = [node._id for node in getFuncNodeInTestID(db, testID)]

    list_all_delete_node = []
    for func_id in list_all_funcID:#allfile in a testID
        list_all_delete_node += getDeleteNode(db, func_id)

    if list_all_delete_node == []:
        return False
    else:
        return list_all_delete_node


def getDeclNode(db, func_id):
    query_str = "queryNodeIndex('type:IdentifierDeclStatement AND functionId:%d')" % func_id
    results = db.runGremlinQuery(query_str)
    return results


def get_all_iddecl_node(db, testID):
    list_all_funcID = [node._id for node in getFuncNodeInTestID(db, testID)]

    list_all_decl_node = []
    for func_id in list_all_funcID:#allfile in a testID
        list_all_decl_node += getDeclNode(db, func_id)

    if list_all_decl_node == []:
        return False
    else:
        return list_all_decl_node


def getCallGraph(db, testID):
    #构建函数的调用关系图
    list_all_func_node = getFuncNodeInTestID(db, testID)
    # print "list_all_func_node: ", list_all_func_node
    # if list_all_func_node == []:
    #     return False
    if list_all_func_node == False:
        return [],{}
    
    call_g = Graph(directed=True)

    for func_node in list_all_func_node:
        #print(func_node)
        prop = {'funcname':func_node.properties['name'], 'type': func_node.properties['type'], 'filepath': func_node.properties['filepath']}
        call_g.add_vertex(str(func_node._id), **prop)


    list_all_callee,func_pointer_dict = get_all_calls_node(db, testID)#we must limit result in testID, it already get callee node
    # print 'list_all_callee: ', list_all_callee
    if list_all_callee == False:
        return [],{}

    for func_node in list_all_func_node:
        function_name = func_node.properties['name']
        # print "function_name", function_name
        tag = False
        if function_name.find('::') != -1: #if is a function in class, have two problems
            func_name = function_name.split('::')[-1].strip()
            classname = function_name.split('::')[0].strip()

            if func_name == classname: #is a class::class, is a statementnode or a iddeclnode
                # print (1)
                list_callee_id = []
                list_delete_node = get_all_delete_node(db, testID)  #delete object
                # print "list_delete_node: ",list_delete_node
                if list_delete_node == False:
                    continue

                for node in list_delete_node:
                    functionID = node.properties["functionId"]
                    all_cfg_node = getCFGNodes(db, functionID)
                    delete_loc = node.properties['location'].split(':')[0]

                    for cfg_node in all_cfg_node:
                        if cfg_node.properties['location'] != None and cfg_node.properties['location'].split(':')[0] == delete_loc and cfg_node.properties['code'] != 'delete' and cfg_node.properties['code'] != '[' and cfg_node.properties['code'] != '[':
                            objectname = cfg_node.properties['code']
                            ob_classname = getClassByObjectAndFuncID(db, objectname, functionID)
                            pdg = getFuncPDGByfuncIDAndtestID(functionID, testID)
                            if pdg == False:
                                continue

                            if ob_classname == classname:
                                # print str(node._id), str(cfg_node._id)
                                for p_n in pdg.vs:
                                    # print p_n['name']
                                    if p_n['name'] == str(node._id):
                                        list_s = p_n.predecessors()   # edge from init node to delete node
                                        for edge in pdg.es:
                                            if pdg.vs[edge.tuple[0]] in list_s and pdg.vs[edge.tuple[1]] == p_n and edge['var'] == objectname:
                                                if str(functionID) == str(pdg.vs[edge.tuple[0]]['name']):
                                                    continue
                                                list_callee_id.append((str(functionID), str(pdg.vs[edge.tuple[0]]['name'])))
                                            else:
                                                continue 

                                    elif p_n['name'] == str(cfg_node._id):
                                        list_s = p_n.predecessors()
                                        for edge in pdg.es:
                                            if pdg.vs[edge.tuple[0]] in list_s and pdg.vs[edge.tuple[1]] == p_n and edge['var'] == objectname:
                                                list_callee_id.append((functionID, str(pdg.vs[edge.tuple[0]]['name'])))
                                            else:
                                                continue  

                        else:
                            continue


                    else:
                        continue

            elif func_name.replace('~', '') == classname:#is a class::~class
                # print (2)
                list_callee_id = []
                list_delete_node = get_all_delete_node(db, testID)
                if list_delete_node == False:
                    continue

                for node in list_delete_node:
                    functionID = node.properties["functionId"]
                    all_cfg_node = getCFGNodes(db, functionID)
                    delete_loc = node.properties['location'].split(':')[0]

                    for cfg_node in all_cfg_node:
                        if cfg_node.properties['location'] != None and cfg_node.properties['location'].split(':')[0] == delete_loc and cfg_node.properties['code'] != 'delete' and cfg_node.properties['code'] != '[' and cfg_node.properties['code'] != '[':
                            objectname = cfg_node.properties['code']
                            #print objectname

                            ob_classname = getClassByObjectAndFuncID(db, objectname, functionID)

                            if ob_classname == classname:
                                pdg = getFuncPDGByfuncIDAndtestID(functionID, testID)
                                if pdg == False:
                                    continue

                                for p_n in pdg.vs:
                                    if p_n['name'] == str(node._id):
                                        list_callee_id.append((functionID, str(node._id)))

                                    elif p_n['name'] == str(cfg_node._id):
                                        list_callee_id.append((functionID, str(cfg_node._id))) #delete and its object node

                        else:
                            continue


                    else:
                        continue

            else:
                # print (3)
                tag = 'func'
                list_callee_id = []
                for _t in list_all_callee:#_t is a tuple, _t[0] is nodeid, 1 is funcname, 2 is func_id
                    if _t[1].find('-> '+ func_name) != -1: #maybe is a class->funcname()
                        objectname = _t[1].split(' -> '+ func_name)[0].strip()
                        ob_classname = getClassByObjectAndFuncID(db, objectname, _t[2])

                        if ob_classname == classname:
                            list_callee_id.append(_t[0])

                        else:
                            continue
                    elif _t[1].find('. '+ func_name) != -1: #maybe is a object.funcname()
                        objectname = _t[1].split(' . '+ func_name)[0].strip()
                        ob_classname = getClassByObjectAndFuncID(db, objectname, _t[2])
                        # print objectname,ob_classname,classname

                        if ob_classname == classname:
                            list_callee_id.append(_t[0])

                        else:
                            continue
                        
                    else:
                        continue


        else:
            tag = 'func'
            list_callee_id = []
            for _t in list_all_callee:
                if _t[1] == function_name:
                    list_callee_id.append(_t[0])

        #print 4, list_callee_id
        if list_callee_id == []:
            continue

        else:
            #change ast node to cfgnode
            list_callee_CFGNode = []
            if tag == 'func':
                for node_id in list_callee_id:
                    callee_cfgnode = getCFGNodeByCallee(db, node_id)

                    if callee_cfgnode == None:
                                                
                        print ('ERROR', callee_cfgnode)
                        continue
                    else:
                        list_callee_CFGNode.append(callee_cfgnode)

                for node in list_callee_CFGNode:
                    startNode = str(node.properties['functionId'])
                    endNode = str(func_node._id)
                    var = str(node._id)
                    call_g = addDataEdge(call_g, startNode, endNode, var)#var is callee node id
            else:
                    startNode = str(node[0])
                    endNode = str(func_node._id)
                    var = str(node[1])
                    call_g = addDataEdge(call_g, startNode, endNode, var)#var is callee node id


    return call_g, func_pointer_dict


def get_func_relation():
    j = JoernSteps()
    j.connectToDatabase()

    pdg_db_path = "pdg_db"
    list_testID = os.listdir(pdg_db_path)
    i=1
    len1=len(list_testID)
    for testID in list_testID:
        print('\r',end='')
        print('callg:',i,'/',len1,' ',end='')
        i+=1
        if os.path.exists(os.path.join("dict_call2cfgNodeID_funcID", str(testID))):
            continue
        # if testID in ['CVE-2013-2212','CVE-2013-1918']:
        #     continue
        # print testID

        call_g,func_pointer_dict = getCallGraph(j, testID)
        if call_g == []:
            continue

        _dict = {}
        for edge in call_g.es:
            endnode = call_g.vs[edge.tuple[1]]

            if endnode['name'] not in _dict:
                _dict[endnode['name']] = [(edge['var'], call_g.vs[edge.tuple[0]]['name'])]

            else:
                _dict[endnode['name']].append((edge['var'], call_g.vs[edge.tuple[0]]['name']))

        if not os.path.exists(os.path.join("dict_call2cfgNodeID_funcID", str(testID))):
            os.makedirs(os.path.join("dict_call2cfgNodeID_funcID", str(testID)))

        filepath = os.path.join("dict_call2cfgNodeID_funcID", str(testID), "func_pointer_dict.pkl")
        f = open(filepath, 'wb')
        pickle.dump(func_pointer_dict, f, True)
        f.close()

        filepath = os.path.join("dict_call2cfgNodeID_funcID", str(testID), "dict.pkl")
        # print _dict
        f = open(filepath, 'wb')
        pickle.dump(_dict, f, True)
        f.close()

if __name__ == "__main__":
    get_func_relation()

