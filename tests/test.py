from stopwatch import Timer
import time

def test_1_timer_output_format():

    # Creating the timer
    timer = Timer()

    timer.start("tag")
    timer.stop()

    dict_tree = timer.as_dict()
        
    for k in [
        "tag",
        "timestamp_start",
        "timestamp_stop",
        "memory_start",
        "memory_stop"
    ]:
        assert k in dict_tree, f"Missing key '{k}' in timer dictionary serialization."

def test_2_timer_core_functionality():

    # create timer
    timer = Timer()

    # create run tag and sleep for half a second
    timer.start("run")
    time.sleep(0.5)
    timer.stop()

    # check that everything is clean
    assert len(timer.stack) == 0
    dict_tree = timer.as_dict()
    assert dict_tree["tag"] == "run"
    assert "children" not in dict_tree or len(dict_tree["children"]) == 0


def test_3_context_manager():
    delay = lambda: time.sleep(0.5)

    # Creating the timer
    timer = Timer()

    # Timer can be used explicitly through the start/stop methods
    run_timer_id = timer.start("run")

    # Timer can be used through context manager (recommended)
    with timer.time("load_data"):
        delay()

    # Timer can catch and cancel branch if error is raised
    with timer.time("fit", cancel_on_error=True) as fit_timer:
        # Simulation of a training loop
        for epoch_i in range(10):
            with timer.time("epoch", {"i": epoch_i}) as epoch_timer:
                delay()

                if epoch_i > 2:
                    # Possible failure for whichever reason
                    raise RuntimeError

                # We can record information for the current node
                # For the time to be accurate it must be done before at the end of the block
                epoch_timer["accuracy"] = 0.5

    # Here the current active node should me the "run"
    assert timer.active_node.id == run_timer_id

    with timer.time("predict") as predict_timer:
        delay()

        # Here is how I can record my metrics through the timer
        predict_timer["train"] = {"accuracy": 1.0}

    timer.stop()

    assert len(timer.stack) == 0
    dict_tree = timer.as_dict()
    assert dict_tree["tag"] == "run"
    assert len(dict_tree["children"]) == 3
