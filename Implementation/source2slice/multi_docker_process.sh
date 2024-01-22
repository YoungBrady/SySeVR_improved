
if [[ $1 == 'start' ]];then
    for num in {0..19}
    #num=19
    do
        # docker run -itd --name sysevr_wh$num -v /home/nfs/m2022-wh/SySeVR:/home/sysevr tk1037/sagpool:1.0 /bin/bash
        tmux new -d -s wh_$num -n window0
        docker restart sysevr_wh$num
        sleep 2
        tmux send -t wh_$num "docker exec sysevr_wh$num /bin/bash -c \"cd /home/sysevr/Implementation/source2slice && ./slice_nvd.sh s $num \"" ENTER
    done
elif [[ $1 == 'stop' ]];then
    for num in {0..19}
    do  
        if [[ $num == 6 || $num == 19 ]];then
            

        tmux kill-session -t wh_$num
        docker stop sysevr_wh$num
        docker rm sysevr_wh$num
        fi
        # docker restart sysevr_wh$num
    done
fi