import os

class Logger:
    """Logger pipeline for managing losses and metrics during training and validation."""
    def __init__(self):
        if not os.path.exists("output/logs"):
            os.makedirs("output/logs")
        self.training = open("output/logs/training_loss.csv", "w")
        self.training.write("epoch,loss\n")

        self.validation = open("output/logs/validation_loss.csv", "w")
        self.validation.write("epoch,loss\n")
    
    def log_training_loss(self, epoch, loss):
        self.training.write(str(epoch) + "," + str(loss) + "\n")

    def log_validation_loss(self, epoch, loss):
        self.validation.write(str(epoch) + "," + str(loss) + "\n")
