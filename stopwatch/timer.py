import time
from contextlib import contextmanager
from typing import Any, Hashable

import numpy as np
import psutil
import os


# !Private class to be used within the Timer class
class TimerNode:
    STARTED: str = "STARTED"
    STOPPED: str = "STOPPED"
    CANCELED: str = "CANCELED"

    def __init__(
        self, id_: int, tag: str, metadata: dict = None, precision: int = 6, timestamp_start=None, memory=None
    ) -> None:
        self.id = id_

        self.precision = precision

        self.tag = tag
        self.status = TimerNode.STARTED
        self.cancellation_source_id = None

        self.timestamp_start = self.time() if timestamp_start is None else timestamp_start
        self.timestamp_end = None
        self.memory_start = memory
        self.memory_end = None

        assert metadata is None or isinstance(metadata, dict)
        self.metadata = {} if metadata is None else metadata

        self.children = []

    def time(self) -> float:
        # return np.round(time.time(), decimals=self.precision)
        return time.time()

    def stop(self, metadata=None, timestamp_end=None, memory=None):
        self.timestamp_end = self.time() if timestamp_end is None else timestamp_end
        self.memory_end = memory
        if metadata is not None:
            self.metadata.update(metadata)
        self.status = TimerNode.STOPPED

        if self.children:
            for i, child in enumerate(self.children[1:], start=1):
                predecessor = self.children[i-1]
                assert child.timestamp_start >= predecessor.timestamp_end, \
                    f"timing error in children of {self.tag}."\
                    f" child node {child.tag} starts at {child.timestamp_start},"\
                    f" which is {predecessor.timestamp_end - child.timestamp_start}s earlier than predecessing child {predecessor.tag} ends ({predecessor.timestamp_end})."
            assert self.timestamp_end >= self.children[-1].timestamp_end, \
                f"node {self.tag} ended {self.children[-1].timestamp_end - self.timestamp_end}s earlier than its last child {self.children[-1].tag}"

    def cancel(self, memory=None):
        self.timestamp_end = self.time()
        self.status = TimerNode.CANCELED
        self.memory_end = memory

    def __getitem__(self, key):
        return self.metadata[key]

    def __setitem__(self, key, value):
        self.metadata[key] = value

    def as_dict(self, timestamp_offset: float = 0) -> dict:
        out = dict(
            # id=self.id,
            tag=self.tag,
            timestamp_start=np.round(
                self.timestamp_start - timestamp_offset, self.precision
            ),
            timestamp_stop=np.round(
                self.timestamp_end - timestamp_offset, self.precision
            )
        )
        if self.memory_start is not None:
            out["memory_start"] = self.memory_start
        if self.memory_end is not None:
            out["memory_stop"] = self.memory_end

        if len(self.metadata) > 0:
            out["metadata"] = self.metadata

        if len(self.children) > 0:
            out["children"] = [c.as_dict(timestamp_offset) for c in self.children]

        if self.cancellation_source_id:
            out["cancellation_source_id"] = self.cancellation_source_id

        return out

    def __repr__(self) -> str:
        return f"TimerNode(id={self.id}, tag={self.tag}, status={self.status})"


class Timer:
    """Class representing a timing profiler.

    Example use:

    >>> timer = Timer()
    >>> timer.start("program")
    >>> timer.start("function_call")
    >>> result = foo()
    >>> timer.stop()
    >>> timer.stop()
    >>> time.as_dict()

    Args:
        precision (int): Number of digits recorded in measurement.
    """

    def __init__(self, precision: int = 6, track_memory=True):
        self.root = None
        self.stack = []
        self.precision = precision
        self.id_counter = 0
        self.accumulated_synthetic_time = 0
        self.track_memory = track_memory
        self.process = psutil.Process(os.getpid()) if track_memory else None
    
    def inject_synthetically_elapsed_time(self, elapsed_time_in_s):
        self.accumulated_synthetic_time += elapsed_time_in_s

    def start(self, tag: Hashable, metadata: dict = None) -> int:
        """Start the timer for a new tag (i.e., creates a child node in the time tree).

        Args:
            tag (Hashable): tag of the node.
            metadata (dict, optional): optional metadata of the node. Defaults to ``None``.
            timestamp_start (int, optional): the timestamp that should be used to tag the start of the node.

        Returns:
            int: id of the created node in the timer tree.
        """

        node = TimerNode(
            self.id_counter,
            tag,
            metadata,
            self.precision,
            timestamp_start=time.time() + self.accumulated_synthetic_time, 
            memory=np.round(self.process.memory_info().rss / 1024**3, 1) if self.track_memory else None
        )
        self.id_counter += 1

        if self.root is None:
            self.root = node
        else:
            parent = self.stack[-1]
            parent.children.append(node)

        self.stack.append(node)

        return node.id

    def stop(self, metadata: dict = None):
        """Stops the current timer and steps back to the parent node"""

        if len(self.stack) == 0:
            raise ValueError("No timer currently active!")

        node = self.stack.pop()
        memory = np.round(self.process.memory_info().rss / 1024**3, 1) if self.track_memory else None
        node.stop(metadata, timestamp_end=time.time() + self.accumulated_synthetic_time, memory=memory)

    def inject(self, timer_node, offset=0, ignore_root=True):
        """
            Injects the full content of another timer inside this timer
        """
        if not ignore_root:
            self.start(tag=timer_node.tag, metadata=timer_node.metadata, timestamp_start=timer_node.timestamp_start + offset)
        for child in timer_node.children:
            self.inject(child, ignore_root=False, offset=offset)
        if not ignore_root:
            self.stop(timestamp_end=timer_node.timestamp_end + offset)


    def cancel(self, node_id: int, only_children: bool = False):
        """Cancels all child nodes up to the node corresponding to node_id (i.e., cancel all branches starting from `node_id`).

        Args:
            node_id (int): id of the node to cancel.
            only_children (bool, optional): if True, only the children of the node are canceled. Defaults to ``False``.
        """

        # Node must be in current set of active nodes
        if node_id not in [n.id for n in self.stack]:
            raise ValueError(
                f"The node timer with id '{node_id}' cannot be canceled because it is not currently active."
            )

        # Active node when cancel was requested
        source = self.active_node

        # Cancel nodes of the branch
        node = None
        while node is None or node.id != node_id:
            node = self.stack.pop()

            if node.id == node_id and only_children:
                self.stack.append(node)
                break

            node.cancel(memory=np.round(self.process.memory_info().rss / 1024**3, 1) if self.track_memory else None)

        # Record source of cancellation at root of cancelled branch
        node.cancellation_source_id = source.id

    @property
    def active_node(self):
        """The current active timer node."""
        if len(self.stack) == 0:
            raise ValueError("No timer currently active!")
        return self.stack[-1]

    def as_dict(self):
        return self.root.as_dict(timestamp_offset=self.root.timestamp_start)

    def as_json(self):
        """
            returns JSON string for the time content
        """
        return lcdb.json.dumps(self.as_dict())

    @contextmanager
    def time(
            self,
            tag: Hashable,
            metadata: dict = None,
            cancel_on_error=False
        ):
        node_id = self.start(tag, metadata)
        try:
            yield self.active_node
        except:
            if cancel_on_error:
                self.cancel(node_id)
            else:
                raise
        else:
            self.stop()
