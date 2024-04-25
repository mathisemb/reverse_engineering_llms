#!/bin/bash

usage() {
    echo "Usage: $0 -h <host_option>"
    exit 1
}

while getopts ":h:" opt; do
    case ${opt} in
        h)
            host_option=$OPTARG
            ;;
        \?)
            echo "Invalid option: -$OPTARG"
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument."
            usage
            ;;
    esac
done

if [ -z "$host_option" ]; then
    usage
fi

case "$host_option" in
    lam)
        host="lam"
        destination_path="/home/lamsade/membit/projects/reverse"
        ;;
    *)
        echo "Invalid host option: $host_option"
        usage
        ;;
esac

rsync -e ssh -avz --filter=':- .rsyncignore' ./ "$host":$destination_path 