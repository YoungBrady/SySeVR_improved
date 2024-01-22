# -*- coding:utf-8 -*-
import pickle
import re
import os
import json
from tqdm import tqdm

def make_label(data_path,label_path,_dict):	
	_labels = {}
	total_len=0
	for type in os.listdir(data_path):
		for batch in os.listdir(os.path.join(data_path,type)):
			for software in os.listdir(os.path.join(data_path,type,batch)):
				for cve in os.listdir(os.path.join(data_path,type,batch,software)):
					if not os.path.exists(os.path.join(data_path,type,batch,software,cve,'slice_source')):
						continue
					for filename in os.listdir(os.path.join(data_path,type,batch,software,cve,'slice_source')):
						total_len+=1
	pbar=tqdm(total=total_len)
	for type in os.listdir(data_path):
		for batch in os.listdir(os.path.join(data_path,type)):
			for software in os.listdir(os.path.join(data_path,type,batch)):
				for cve in os.listdir(os.path.join(data_path,type,batch,software)):
					if not os.path.exists(os.path.join(data_path,type,batch,software,cve,'slice_source')):
						continue
					for filename in os.listdir(os.path.join(data_path,type,batch,software,cve,'slice_source')):
							pbar.update(1)
						# try:
							filepath = os.path.join(data_path,type,batch,software,cve,'slice_source',filename)
							
							f = open(filepath,'r')
							slicelists = f.read().split('------------------------------')
							f.close()

							# labelpath = os.path.join(label_path,filename[:-4]+'_label.pkl')	

							if slicelists[0] == '':
								del slicelists[0]
							if slicelists[-1] == '' or slicelists[-1] == '\n' or slicelists[-1] == '\r\n':
								del slicelists[-1]
						
							for slice in slicelists:
								# try:
									sentences = slice.split('\n')
									if sentences[0] == '\r' or sentences[0] == '':
										del sentences[0]
									if sentences == []:
										continue
									if sentences[-1] == '':
										del sentences[-1]
									if sentences[-1] == '\r':
										del sentences[-1]
								
									slicename = sentences[0]
									label = 0
									try:
										key = '../../Program_data/file/' + ('/').join(slicename.split(' ')[1].split('/')[-3:])  #key in label_source
									except:
										continue

									if key not in _dict.keys():
										_labels[slicename] = 0
										continue
									if len(_dict[key]) == 0:
										_labels[slicename] = 0
										continue
									sentences = sentences[1:]
									for sentence in sentences:
										# if (is_number(sentence.split(' ')[-1])) is False:
										# 	continue
										try:
											linenum = int(sentence.split(' ')[-1])
										except:
											continue
										vullines = _dict[key]
										if linenum in vullines:
											label = 1
											_labels[slicename] = 1
											break 
									if label == 0:
										_labels[slicename] = 0	
								# except Exception as e:
								# 	print(e)
								# 	print(key)
								# 	continue
						# except Exception as e:
						# 	print(e)
						# 	continue
	os.makedirs(label_path,exist_ok=True)
	labelpath=os.path.join(label_path,'labels.json')
	with open(labelpath,'w+') as f1:
		json.dump(_labels,f1)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False	
	
if __name__ == '__main__':
	# os.chdir('/home/sysevr/Implementation/source2slice')
	with open('./vulline_dict.json','r') as f:
		_dict = json.load(f)

	#print(_dict)

	code_path = '../../slice_all'  #slice code of software
	label_path = '../labels'   #labels
	
	make_label(code_path,label_path,_dict)	
