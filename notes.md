# Training

## Accessing the Remote Machine
I open a session with the remote using ssh

    ssh -L 6006:localhost:6006 nathan@129.97.68.241

I then enter my account password

    xxxx
## Sourcing the virtual environment
On the remote machine I source my virtual environment for reinforcement learning *pyrl2*

    source pyrl2/bin/activate
## Running the program
To run the navigate I first first move to the desired directory

    cd PyFlyt

And ensure I have the latest version

    git pull

If I do need to modify the scripts, because we are in a terminal, I like to use nano

    nano script.py

When ready I launch the program with something like

    nohup python3 test_script.py --ARGS arg1 > output.log 2>&1 &

which will run the program in the background and send output statements to *output.log*

## Monitoring
I can check on the program's status with

    ps aux | grep test_sa_envs.py

Or view the output with

    tail -f output.log

I can also view the tensorboard logs by navigating to their event file's location and running

    tensorboard --logdir=. --port=6006

Then on my local PC I can view the tensorboard at *http://localhost:6006*
