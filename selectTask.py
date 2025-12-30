def select_task():
    """
    Allow the user to choose the type of motor imagery task for EEG analysis.
    The chosen task will determine which experimental runs are loaded and processed.
    """
    TASK_RUNS = {
        "hands_vs_feet": [6, 10, 14],
        "left_vs_right": [4, 8, 12],
        "rest_vs_movement": [1]
    }

    TASK_DESCRIPTIONS = {
        "hands_vs_feet": "Differentiate between imagining hand movements and foot movements.",
        "left_vs_right": "Differentiate between imagining left-hand vs right-hand movements.",
        "rest_vs_movement": "Differentiate between rest (no movement) and imagining movement."
    }

    print("Task options for EEG analysis:")
    for i, key in enumerate(TASK_RUNS.keys()):
        print(f"{i+1}. {key} - {TASK_DESCRIPTIONS[key]}")

    choice = int(input("Please choose the task (enter the number): ")) - 1
    task_name = list(TASK_RUNS.keys())[choice]
    runs = TASK_RUNS[task_name]
    return task_name, runs
