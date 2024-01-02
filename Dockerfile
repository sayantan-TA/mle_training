FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive

# Update package list and install necessary packages
RUN apt-get update && \
    apt-get install -y python3 python3-pip wget && \
    rm -rf /var/lib/apt/lists/*

# Create a directory for the application
RUN mkdir /home/mle-training

# Copy the content of the current directory to /home/mle-training
COPY . /home/mle-training

# Make the script executable
RUN chmod +x /home/mle-training/commands.sh

# Set the working directory
WORKDIR /home/mle-training

# Use conda run to execute commands within the Conda environment
CMD ["./commands.sh"]

