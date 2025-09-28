import time

def slow_task(name, delay):
    print(f"Starting task {name}: {time.strftime('%X')}")
    time.sleep(delay)
    print(f"Finished task {name}: {time.strftime('%X')}")

def main():
    print(f"Program started: {time.strftime('%X')}")
    slow_task("Task 1", 3)
    slow_task("Task 2", 7)
    print(f"Program finished: {time.strftime('%X')}")
    print(f"Total time expected: 10 seconds")

main()