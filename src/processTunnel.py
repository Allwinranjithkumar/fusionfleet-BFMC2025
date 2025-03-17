if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../../..")

from src.templates.workerprocess import WorkerProcess
from src.Keeper.Lane.threads.threadTunnel import threadTunnel

class processLane(WorkerProcess):
    """This process handles Lane.
    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        debugging (bool, optional): A flag for debugging. Defaults to False.
    """

    def __init__(self, queueList, logging, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        self.debugging = debugging
        super(processRamp, self).__init__(self.queuesList)

    def run(self):
        """Apply the initializing methods and start the threads."""
        super(processTunnel, self).run()

    def _init_threads(self):
        """Create the Lane Publisher thread and add to the list of threads."""
        TunnelTh= threadTunnel(
            self.queuesList, self.logging, self.debugging
        )
        self.threads.append(TunnelTh)

