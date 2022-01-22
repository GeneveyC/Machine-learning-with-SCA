#! /bin/sh

if [ $# -ne 0 ]; then
    port=$1
else
    port=8888
fi

# look for a free port
free=`sudo ss -tulpn | grep -c -e ":$port"`
while [ $free -ne 0 ]; do
    port=$((port + 1))	
    free=`sudo ss -tulpn | grep -c -e ":$port"`
done

sudo iptables -I INPUT 1 -p tcp --dport $port -j ACCEPT
#unset XDG_RUNTIME_DIR

jupyter notebook --no-browser --NotebookApp.allow_origin="*" --port="$port" --ip="0.0.0.0" --notebook-dir="."
