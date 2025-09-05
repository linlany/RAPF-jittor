import abc
import enum
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from torchvision import transforms

# from continuum.datasets import _ContinuumDataset
# from continuum.tasks import TaskSet, TaskType
# from continuum.transforms.segmentation import Compose as SegmentationCompose

from PIL import Image
from jittor.dataset import Dataset
import jittor as jt
from jittor.dataset import ImageFolder

def _handle_negative_indexes(index: int, total_len: int) -> int:
    if index < 0:
        index = index % total_len
    return index


class transforms_seg:
    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, img, target):
            for t in self.transforms:
                img, target = t(img, target)
            return img, target

    class ToTensor:
        def __call__(self, img, target):
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            if not isinstance(target, Image.Image):
                target = Image.fromarray(target)
            
            return transforms.to_tensor(img), jt.array(np.array(target)).long()

class TaskType(enum.Enum):
    """Enumeration to list all possible data types supported."""
    IMAGE_ARRAY = 1
    IMAGE_PATH = 2
    TEXT = 3
    TENSOR = 4
    SEGMENTATION = 5
    OBJ_DETECTION = 6
    H5 = 7
    AUDIO = 8

class TaskSet(Dataset):
    def __init__(self, x, y, t, trsf=None, target_trsf=None, data_type=TaskType.IMAGE_ARRAY, bounding_boxes=None, data_indexes=None):
        super().__init__()
        self._x, self._y, self._t = x, y, t
        self.trsf = trsf
        self.target_trsf = target_trsf
        self.data_type = data_type
        self.bounding_boxes = bounding_boxes
        self.data_indexes = data_indexes
        
        self.set_attrs(total_len=len(self._x))

    def __getitem__(self, index):
        x, y, t = self.x[index], self.y[index], self.t[index]

        if self.data_type == TaskType.IMAGE_PATH:
            if isinstance(x, bytes):
                x = x.decode('utf-8')
            img = Image.open(x).convert('RGB')
        elif self.data_type == TaskType.IMAGE_ARRAY:
            img = Image.fromarray(x)
        else:
            img = x

        if self.trsf:
            img = self.trsf(img)

        if self.target_trsf:
            y = self.target_trsf(y)

        return img, y, t

def _slice(
    y: np.ndarray,
    t: Optional[np.ndarray],
    keep_classes: Optional[List[int]] = None,
    discard_classes: Optional[List[int]] = None,
    keep_tasks: Optional[List[int]] = None,
    discard_tasks: Optional[List[int]] = None
):
    """Slice dataset to keep/discard some classes/task-ids.

    Note that keep_* and and discard_* are mutually exclusive.
    Note also that if a selection (keep or discard) is being made on the classes
    and on the task ids, the resulting intersection will be taken.

    :param y: An array of class ids.
    :param t: An array of task ids.
    :param keep_classes: Only keep samples with these classes.
    :param discard_classes: Discard samples with these classes.
    :param keep_tasks: Only keep samples with these task ids.
    :param discard_tasks: Discard samples with these task ids.
    :return: A new Continuum dataset ready to be given to a scenario.
    """
    if keep_classes is not None and discard_classes is not None:
        raise ValueError("Only use `keep_classes` or `discard_classes`, not both.")
    if keep_tasks is not None and discard_tasks is not None:
        raise ValueError("Only use `keep_tasks` or `discard_tasks`, not both.")

    if t is None and (keep_tasks is not None or discard_tasks is not None):
        raise Exception(
            "No task ids information is present by default with this dataset, "
            "thus you cannot slice some task ids."
        )
    y = y.astype(np.int64)
    if t is not None:
        t = t.astype(np.int64)

    indexes = set()
    if keep_classes is not None:
        indexes = set(np.where(np.isin(y, keep_classes))[0])
    elif discard_classes is not None:
        keep_classes = list(set(y) - set(discard_classes))
        indexes = set(np.where(np.isin(y, keep_classes))[0])

    if keep_tasks is not None:
        _indexes = np.where(np.isin(t, keep_tasks))[0]
        if len(indexes) > 0:
            indexes = indexes.intersection(_indexes)
        else:
            indexes = indexes.union(_indexes)
    elif discard_tasks is not None:
        keep_tasks = list(set(t) - set(discard_tasks))
        _indexes = np.where(np.isin(t, keep_tasks))[0]
        if len(indexes) > 0:
            indexes = indexes.intersection(_indexes)
        else:
            indexes = indexes.union(_indexes)

    indexes = np.array(list(indexes), dtype=np.int64)
    return indexes

class _ContinuumDataset(Dataset):

    def __init__(self, data_path: str = "", train: bool = True, download: bool = True) -> None:
        self.data_path = os.path.expanduser(data_path) if data_path is not None else None
        self.download = download
        self.train = train

        if self.data_path is not None and self.data_path != "" and not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        if self.download:
            self._download()

        if not isinstance(self.data_type, TaskType):
            raise NotImplementedError(
                f"Dataset's data_type ({self.data_type}) is not supported."
                " It must be a member of the enum TaskType."
            )

        # Initialization of the default properties
        if self.data_type == TaskType.SEGMENTATION:
            self._trsf = [transforms_seg.ToTensor()]
        else:
            self._trsf = [transforms.ToTensor()]
        self._bboxes = None
        self._attributes = None

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the loaded data under the form of x, y, and t."""
        raise NotImplementedError("This method should be implemented!")

    def _download(self):
        pass

    def slice(
            self,
            keep_classes: Optional[List[int]] = None,
            discard_classes: Optional[List[int]] = None,
            keep_tasks: Optional[List[int]] = None,
            discard_tasks: Optional[List[int]] = None
    ):
        """Slice dataset to keep/discard some classes/task-ids.

        Note that keep_* and and discard_* are mutually exclusive.
        Note also that if a selection (keep or discard) is being made on the classes
        and on the task ids, the resulting intersection will be taken.

        :param keep_classes: Only keep samples with these classes.
        :param discard_classes: Discard samples with these classes.
        :param keep_tasks: Only keep samples with these task ids.
        :param discard_tasks: Discard samples with these task ids.
        :return: A new Continuum dataset ready to be given to a scenario.
        """
        if self.data_type == TaskType.SEGMENTATION:
            raise NotImplementedError("It's not possible yet to slice Segmentation datasets.")

        x, y, t = self.get_data()

        indexes = _slice(
            y, t,
            keep_classes, discard_classes,
            keep_tasks, discard_tasks
        )

        new_x, new_y, new_t = x[indexes], y[indexes], None
        if t is not None:
            new_t = t[indexes]
        sliced_dataset = InMemoryDataset(
            new_x, new_y, new_t,
            data_type=self.data_type
        )
        sliced_dataset.attributes = self.attributes
        sliced_dataset.bounding_boxes = self.bounding_boxes
        sliced_dataset.transformations = self.transformations

        return sliced_dataset

    @property
    def nb_classes(self) -> List[int]:
        return None

    @property
    def class_order(self) -> Union[None, List[int]]:
        return None

    @property
    def need_class_remapping(self) -> bool:
        """Flag for method `class_remapping`."""
        return False

    def class_remapping(self, class_ids: np.ndarray) -> np.ndarray:
        """Optional class remapping.

        Used for example in PermutedMNIST, cf transformed.py;

        :param class_ids: Original class_ids.
        :return: A remapping of the class ids.
        """
        return class_ids

    def to_taskset(
            self,
            trsf: Optional[List[Callable]] = None,
            target_trsf: Optional[List[Callable]] = None
    ) -> TaskSet:
        """Returns a TaskSet that can be directly given to a torch's DataLoader.

        You can use this method if you don't care about the continual aspect and
        simply want to use the datasets in a classical supervised setting.

        :param trsf: List of transformations to be applied on x.
        :param target_trsf: List of transformations to be applied on y.
        :return taskset: A taskset which implement the interface of torch's Dataset.
        """
        if trsf is None and self.data_type == TaskType.SEGMENTATION:
            trsf = transforms_seg.Compose(self.transformations)
        elif trsf is None:
            trsf = transforms.Compose(self.transformations)

        return TaskSet(
            *self.get_data(),
            trsf=trsf,
            target_trsf=target_trsf,
            data_type=self.data_type,
            bounding_boxes=self.bounding_boxes
        )

    @property
    def class_order(self) -> Union[None, List[int]]:
        return None

    @property
    def need_class_remapping(self) -> bool:
        """Flag for method `class_remapping`."""
        return False

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_ARRAY

    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        return self._trsf

    @transformations.setter
    def transformations(self, trsf: List[Callable]):
        self._trsf = trsf

    @property
    def bounding_boxes(self) -> List:
        """Returns a bounding box (x1, y1, x2, y2) per sample if they need to be cropped."""
        return self._bboxes

    @bounding_boxes.setter
    def bounding_boxes(self, bboxes: List):
        self._bboxes = bboxes

    @property
    def attributes(self) -> np.ndarray:
        """Returns normalized attributes for all class if available.

        Those attributes can often be found in dataset used for Zeroshot such as
        CUB200, or AwA. The matrix shape is (nb_classes, nb_attributes), and it
        has been L2 normalized along side its attributes dimension.
        """
        return self._attributes

    @attributes.setter
    def attributes(self, attributes: np.ndarray):
        self._attributes = attributes


class InMemoryDataset(_ContinuumDataset):
    """Continuum dataset for in-memory data.

    :param x_train: Numpy array of images or paths to images for the train set.
    :param y_train: Targets for the train set.
    :param data_type: Format of the data.
    :param t_train: Optional task ids for the train set.
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            t: Union[None, np.ndarray] = None,
            data_type: TaskType = TaskType.IMAGE_ARRAY,
            train: bool = True,
            download: bool = True,
    ):
        self._data_type = data_type
        super().__init__(train=train, download=download)

        if len(x) != len(y):
            raise ValueError(f"Number of datapoints ({len(x)}) != number of labels ({len(y)})!")
        if t is not None and len(t) != len(x):
            raise ValueError(f"Number of datapoints ({len(x)}) != number of task ids ({len(t)})!")

        self.data = (x, y, t)
        self._nb_classes = len(np.unique(y))

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.data

    @property
    def nb_classes(self) -> List[int]:
        return self._nb_classes

    @property
    def data_type(self) -> TaskType:
        return self._data_type

    @data_type.setter
    def data_type(self, data_type: TaskType) -> None:
        self._data_type = data_type
class ImageFolderDataset(_ContinuumDataset):
    """Continuum dataset for datasets with tree-like structure.

    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = True,
            data_type: TaskType = TaskType.IMAGE_PATH
    ):
        self.data_path = data_path
        self._data_type = data_type
        super().__init__(data_path=data_path, train=train, download=download)

        allowed_data_types = (TaskType.IMAGE_PATH, TaskType.SEGMENTATION)
        if data_type not in allowed_data_types:
            raise ValueError(f"Invalid data_type={data_type}, allowed={allowed_data_types}.")

    @property
    def data_type(self) -> TaskType:
        return self._data_type

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Union[None, np.ndarray]]:
        self.dataset = ImageFolder(self.data_path)
        x, y, t = self._format(self.dataset.imgs)
        self.list_classes = np.unique(y)
        return x, y, t

    @staticmethod
    def _format(raw_data: List[Tuple[str, int]]) -> Tuple[np.ndarray, np.ndarray, None]:
        x = np.empty(len(raw_data), dtype="S255")
        y = np.empty(len(raw_data), dtype=np.int16)

        for i, (path, target) in enumerate(raw_data):
            x[i] = path
            y[i] = target

        return x, y, None
    

class SegmentationCompose:
    """Composes several transforms together.

    :param transforms: list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl=None):
        if lbl is not None:
            for t in self.transforms:
                img, lbl = t(img, lbl)
            return img, lbl
        else:
            for t in self.transforms:
                img = t(img)
            return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class _BaseScenario(abc.ABC):
    """Abstract loader.

    DO NOT INSTANTIATE THIS CLASS.

    :param cl_dataset: A Continuum dataset.
    :param nb_tasks: The number of tasks to do.
    :param transformations: A list of transformations applied to all tasks. If
                            it's a list of list, then the transformation will be
                            different per task.
    """

    def __init__(
            self,
            cl_dataset: _ContinuumDataset,
            nb_tasks: int,
            transformations: Union[List[Callable], List[List[Callable]]] = None
    ) -> None:

        self.cl_dataset = cl_dataset
        self._nb_tasks = nb_tasks
        self.transformations = transformations
        self._counter = 0

        if transformations is None:
            self.transformations = self.cl_dataset.transformations
        if self.cl_dataset.data_type == TaskType.SEGMENTATION:
            composer = SegmentationCompose
        else:
            composer = transforms.Compose
        if self.transformations is not None and isinstance(self.transformations[0], list):
            # We have list of list of callable, where each sublist is dedicated to
            # a task.
            if len(self.transformations) != nb_tasks:
                raise ValueError(
                    f"When using different transformations per task, there must be as as much transformations"
                    f" ({len(transformations)}) than there are tasks ({nb_tasks})"
                    f", which is not currently the case."
                )
            self.trsf = [composer(trsf) for trsf in self.transformations]
        else:
            self.trsf = composer(self.transformations)

    @abc.abstractmethod
    def _setup(self, nb_tasks: int) -> int:
        raise NotImplementedError

    @property
    def train(self) -> bool:
        """Returns whether we are in training or testing mode.

        This property is dependent on the dataset, not the actual scenario.
        """
        return self.cl_dataset.train

    @property
    def nb_samples(self) -> int:
        """Total number of samples in the whole continual setting."""
        return len(self.dataset[0])  # type: ignore

    @property
    def nb_classes(self) -> int:
        """Total number of classes in the whole continual setting."""
        return len(np.unique(self.dataset[1]))  # type: ignore

    @property
    def classes(self) -> List:
        """list of classes in the whole continual setting."""
        return np.unique(self.dataset[1])  # type: ignore

    @property
    def nb_tasks(self) -> int:
        """Number of tasks in the whole continual setting."""
        return len(self)

    def __len__(self) -> int:
        """Returns the number of tasks.

        :return: Number of tasks.
        """
        return self._nb_tasks

    def __iter__(self):
        """Used for iterating through all tasks with the CLLoader in a for loop."""
        self._counter = 0
        return self

    def __next__(self) -> TaskSet:
        """An iteration/task in the for loop."""
        if self._counter >= len(self):
            raise StopIteration
        task = self[self._counter]
        self._counter += 1
        return task

    def __getitem__(self, task_index: Union[int, slice]):
        """Returns a task by its unique index.

        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A train PyTorch's Datasets.
        """
        if isinstance(task_index, slice) and isinstance(self.trsf, list):
            raise ValueError(
                f"You cannot select multiple task ({task_index}) when you have a "
                "different set of transformations per task"
            )

        x, y, t, _, data_indexes = self._select_data_by_task(task_index)

        return TaskSet(
            x, y, t,
            trsf=self.trsf[task_index] if isinstance(self.trsf, list) else self.trsf,
            data_type=self.cl_dataset.data_type,
            bounding_boxes=self.cl_dataset.bounding_boxes,
            data_indexes=data_indexes
        )

    def _select_data_by_task(
            self,
            task_index: Union[int, slice, np.ndarray]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray, Union[int, List[int]]]:
        """Selects a subset of the whole data for a given task.

        This class returns the "task_index" in addition of the x, y, t data.
        This task index is either an integer or a list of integer when the user
        used a slice. We need this variable when in segmentation to disentangle
        samples with multiple task ids.

        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A tuple of numpy array being resp. (1) the data, (2) the targets,
                 (3) task ids, and (4) the actual task required by the user.
        """

        # conversion of task_index into a list

        if isinstance(task_index, slice):
            start = task_index.start if task_index.start is not None else 0
            stop = task_index.stop if task_index.stop is not None else len(self) + 1
            step = task_index.step if task_index.step is not None else 1
            task_index = list(range(start, stop, step))
            if len(task_index) == 0:
                raise ValueError(f"Invalid slicing resulting in no data (start={start}, end={stop}, step={step}).")

        if isinstance(task_index, np.ndarray):
            task_index = list(task_index)

        x, y, t = self.dataset  # type: ignore

        if isinstance(task_index, list):
            task_index = [
                t if t >= 0 else _handle_negative_indexes(t, len(self)) for t in task_index
            ]
            if len(t.shape) == 2:
                data_indexes = np.unique(np.where(t[:, task_index] == 1)[0])
            else:
                data_indexes = np.where(np.isin(t, task_index))[0]
        else:
            if task_index < 0:
                task_index = _handle_negative_indexes(task_index, len(self))

            if len(t.shape) == 2:
                data_indexes = np.where(t[:, task_index] == 1)[0]
            else:
                data_indexes = np.where(t == task_index)[0]

        if self.cl_dataset.data_type == TaskType.H5:
            # for h5 TaskType, x is just the filename containing all data
            # no need for slicing here
            selected_x = x
        else:
            selected_x = x[data_indexes]
        selected_y = y[data_indexes]
        selected_t = t[data_indexes]

        if self.cl_dataset.need_class_remapping:  # TODO: to remove with TransformIncremental
            # A remapping of the class ids is done to handle some special cases
            # like PermutedMNIST or RotatedMNIST.
            selected_y = self.cl_dataset.class_remapping(selected_y)

        return selected_x, selected_y, selected_t, task_index, data_indexes
