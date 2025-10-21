import re
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def main(log_path: str) -> None:

    with open(log_path, "r") as f:
        log_data = f.read()
        
    train_pattern = re.compile(r"Train, Round\s+(\d+): Loss=([\d\.]+), Accuracy=([\d\.]+)(, gradient_norm=([\d\.]+))?")

    # Regex to extract test metrics
    test_pattern = re.compile(r"Test, Round\s+(\d+): Loss=([\d\.]+), Accuracy=([\d\.]+)")
    rounds, train_losses, train_accuracies, grads = [], [], [], []
    test_rounds, test_losses, test_accuracies = [], [], []

    for match in train_pattern.finditer(log_data):
        rounds.append(int(match.group(1)))
        train_losses.append(float(match.group(2)))
        train_accuracies.append(float(match.group(3)))
        if match.group(4):
            grads.append(float(match.group(5)))


    # Parse test logs
    for match in test_pattern.finditer(log_data):
        test_rounds.append(int(match.group(1)))
        test_losses.append(float(match.group(2)))
        test_accuracies.append(float(match.group(3)))
        
    if rounds == []:
        train_pattern = re.compile(r"Train, Round (\d+) : loss => ([\d\.]+),  accuracy: ([\d\.]+)")
        gradient_norm_pattern = re.compile(r"Train, Round(\d+) : gradient_norm : ([\d\.]+)")
        # Regex to extract test metrics
        test_pattern = re.compile(r"Test, Round (\d+) : loss => ([\d\.]+),  accuracy: ([\d\.]+), std: ([\d\.]+)")

        rounds, train_losses, train_accuracies, grads = [], [], [], []
        test_rounds, test_losses, test_accuracies = [], [], []

        for match in train_pattern.finditer(log_data):
            rounds.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            train_accuracies.append(float(match.group(3)))
    
        for match in gradient_norm_pattern.finditer(log_data):
            grads.append(float(match.group(2)))

        # Parse test logs
        for match in test_pattern.finditer(log_data):
            test_rounds.append(int(match.group(1)))
            test_losses.append(float(match.group(2)))
            test_accuracies.append(float(match.group(3)))
        
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, train_losses, label="Train Loss", color="red")
    if test_losses:
        plt.plot(test_rounds, test_losses, label="Test Loss", color="orange")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Training vs Test Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{os.path.dirname(log_path)}/loss.png', dpi=300)
    plt.close()


    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, np.convolve(train_accuracies, np.ones(10)/10)[:len(rounds)], label="Train Accuracy", color="blue")
    if test_accuracies:
        plt.plot(test_rounds, np.convolve(test_accuracies, np.ones(10)/10)[:len(rounds)], label="Test Accuracy", color="cyan")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Training vs Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{os.path.dirname(log_path)}/accuracy.png', dpi=300)
    plt.close()

    if grads:
        plt.figure(figsize=(8, 5))
        plt.plot(rounds, grads, label="Gradient Norm", color="green")
        plt.xlabel("Round")
        plt.ylabel("Gradient Norm")
        plt.title("Gradient Norm")
        plt.grid(True)
        plt.savefig(f'{os.path.dirname(log_path)}/gradient_norm.png', dpi=300)
        plt.close()
        
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Flower Metrics from Log File")
    parser.add_argument("-l", "--log_path", type=str, required=True, help="Path to the log file")
    args = parser.parse_args()
    
    if not os.path.exists(args.log_path):
        print(f"Le fichier de log {args.log_path} n'existe pas.")
    else:
        main(args.log_path)