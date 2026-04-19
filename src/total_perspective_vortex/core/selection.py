TASK_RUNS = {
    "hands_vs_feet": [6, 10, 14],
    "left_vs_right": [4, 8, 12],
    "rest_vs_movement": [1, 2, 3, 4, 5],
}

TASK_DESCRIPTIONS = {
    "hands_vs_feet": (
        "Differentiate between imagining hand movements and foot movements."
    ),
    "left_vs_right": (
        "Differentiate between imagining left-hand vs right-hand movements."
    ),
    "rest_vs_movement": (
        "Differentiate between rest (no movement) and imagining movement."
    ),
}


def select_task():
    """Prompt the user to select an EEG task and return its runs."""
    task_names = list(TASK_RUNS.keys())

    print("Task options for EEG analysis:")
    for idx, name in enumerate(task_names, start=1):
        print(f"{idx}. {name} - {TASK_DESCRIPTIONS[name]}")

    while True:
        try:
            choice = int(
                input("Please choose the task (enter the number): ")
            )

            if 1 <= choice <= len(task_names):
                task_name = task_names[choice - 1]
                return task_name, TASK_RUNS[task_name]

            print("Invalid choice. Please enter a valid number.")

        except ValueError:
            print("Invalid input. Please enter a number.")
