import subprocess
import os
import time
import signal
import collections
import atexit
import sys

conda_which = str(os.popen('which conda').read())
conda_path = os.path.dirname(os.path.dirname(conda_which))
conda_file = os.path.join(conda_path, 'etc', 'profile.d', 'conda.sh')
conda_env = 'machinelearning'

base_path = os.path.dirname(os.path.abspath(__file__))
executor_file = os.path.join(base_path, 'site_executor.py')
interceptor_file = os.path.join(base_path, 'site_interceptor.py')

def run_python_command(file, port):
    
    commands = [
      "source %s" % (conda_file),
      "conda activate %s" % (conda_env),
      "python %s %i" % (file, port)
    ]

    proc = subprocess.Popen(
      " && ".join(commands),
      shell=True,
      executable="/bin/bash",
      stdout=sys.stdout, 
      stderr=sys.stderr
    )
    
    return proc

def run_executor(port):
    return run_python_command(executor_file, port)

def run_interceptor(port):
    return run_python_command(interceptor_file, port)

def get_stdout(proc):
    return proc.stdout.read().decode('utf8')

def get_stderr(proc):
    return proc.stderr.read().decode('utf8')

def kill_proc(proc):
    proc.terminate()
    outs, errs = proc.communicate()

    try:
        defunct_pid = os.getpgid(proc.pid)
        os.killpg(defunct_pid, signal.SIGTERM)
        print('Process terminated by force.')
    except:
        print('Process terminated normally.')
    
    return

num_sites = 2
interceptors = []
executors = []

for i in range(0, num_sites):
    interceptor_port = 8080 + i * 2 + 1
    executor_port = 8080 + i * 2 + 2

    print('Starting executor process for site %i ...' % i)
    executor_proc = run_executor(executor_port)
    executors.append(executor_proc)
    print('Started executor process.')

    print('Starting interceptor process for site %i ...' % i)
    interceptor_proc = run_interceptor(interceptor_port)
    interceptors.append(interceptor_proc)
    print('Started interceptor process.')

def handle_exit(*args):
    for proc in interceptors:
        kill_proc(proc)
    
    for proc in executors:
        kill_proc(proc)

    print('Exit: cleaned procs.')
    exit()
    
atexit.register(handle_exit)
signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

while True:
    time.sleep(1)
    print('.')