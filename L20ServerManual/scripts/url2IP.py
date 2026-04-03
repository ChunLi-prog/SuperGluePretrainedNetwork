"""
Tools to convert default hostfile on cluster.

For mpi launcher on multi machines.
"""
import socket
import time


def URL2IP():
    for oneurl in urllist.readlines():
        if not oneurl.strip():
            continue
        url = str(oneurl.strip())

        for i in range(10):
            try:
                ip = socket.gethostbyname(url)
                if ip is not None:
                    print(f"Successfully get {ip} by {url}!!!")
                    iplist.writelines(str(ip) + "\n")
                    break
            except Exception as e:
                print(f"{i}-th attempt to get hostip by hostname failed!!!")
                print(e)
                time.sleep(1)
                ip = None
        if ip is None:
            raise ConnectionError(f"{url} to IP ERROR !")


try:
    urllist = open("/job_data/hosts", "r")
    iplist = open("/job_data/mpi_hosts", "w")
    URL2IP()
    urllist.close()
    iplist.close()
    print("complete !")
except Exception as e:
    print("ERROR !")
    raise e
