import os
import sys
import subprocess


def selectGPU():
    ns = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE).stdout.read()
    lns = str(ns).split("\\n")
    gpuCnt = 0

    # count GPUs in system
    for n in lns:
        if "Quadro" in n or "GeForce" in n or "Tesla" in n or "TITAN" in n:
            gpuCnt += 1
        # print(n)

    # find busy GPUs
    gpuInUse = []
    for idx, n in enumerate(lns):
        if "Processes" in n:
            n = lns[idx+1]
            if "GPU" in n and "PID" in n:
                for idx in range(idx+3, len(lns)):
                    if "+----------" in lns[idx]:
                        break
                    ln = lns[idx]
                    pid = ln.split()[1]
                    if pid == "No":
                        break
                    gpuInUse.append(int(pid))

    # find free GPU
    selectedGPU = None
    for g in range(gpuCnt):
        if g not in gpuInUse:
            selectedGPU = g
            break
    return selectedGPU


def main():
    selectGPU()

if __name__ == "__main__":
    main()
