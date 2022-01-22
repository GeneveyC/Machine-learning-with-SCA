
case "$HOSTNAME" in
    "parallel")
        env_path="/home/lab/sca"
        ;;
    "parallel2")
        env_path="/home/lab/venvs/scal_dl"
        ;;
    "parallel3")
        env_path="/home/lab/venv/sca"
        ;;
    "parallel4")
        env_path="/home/lab/venv"
        ;;
     *)
        env_path=""
        ;;
esac

source "$env_path/bin/activate"

# Add to python path
PYTHONPATH=/mnt/SCA1/CARDIS/script:/mnt/SCA1/CARDIS/scripts:/mnt/SCA1/gits/sca_platform:$PYTHONPATH
