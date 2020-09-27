#!/usr/bin/python3

from abc import abstractmethod, ABC
from enum import Enum

import cv2 as cv
import numpy as np
from mypy.types import List


class RecordingState(Enum):
    IDLE = (0, 0, 255)
    RECORDING = (0, 255, 0)


class DataWriter(ABC):
    @abstractmethod
    def write(self, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class VideoWriter(DataWriter, ABC):
    pass


class PaletteWriter(DataWriter, ABC):
    pass


class OpenCVVideoWriter(VideoWriter):
    def __init__(self, filename: str):
        self.writer = cv.VideoWriter()
        fourcc = cv.VideoWriter_fourcc(*'LAGS')
        self.writer.open(filename + '.avi', fourcc, 59.9, (1920, 1080))

    def write(self, image: np.ndarray) -> None:
        self.writer.write(image)

    def close(self) -> None:
        self.writer.release()


class ImageWriter(VideoWriter):
    def __init__(self, filename: str):
        self.filename: str = filename
        self.frame_count: int = 0

    def write(self, data: np.ndarray) -> None:
        cv.imwrite(f'{self.filename}_{self.frame_count}.jpg', data)
        self.frame_count += 1

    def close(self) -> None:
        pass


class SingleFileNumpyWriter(DataWriter):
    def __init__(self, filename):
        self.filename: str = filename
        self.buffer: List[np.array] = []

    def write(self, data: np.ndarray) -> None:
        self.buffer.append(data)

    def close(self) -> None:
        data = np.vstack(self.buffer)
        print(data.shape)
        np.save(self.filename, data, allow_pickle=False)


class PerFrameNumpyWriter(DataWriter):
    def __init__(self, filename: str):
        self.filename: str = filename
        self.frame_count: int = 0

    def write(self, data: np.ndarray) -> None:
        np.save(f'{self.filename}_{self.frame_count}', data, allow_pickle=False)
        self.frame_count += 1

    def close(self) -> None:
        pass


class StateRunInfo:
    def __init__(self, image, palette, is_kayaker_in_image: bool, output_filename_template: str):
        self.image: np.ndarray = image
        self.palette: np.ndarray = palette
        self.is_kayaker_in_image: bool = is_kayaker_in_image
        self.output_filename_template: str = output_filename_template


class InfoRecorder:
    def __init__(self, video_writer: DataWriter, palette_writer: DataWriter):
        self.videoWriter = video_writer
        self.paletteWriter = palette_writer

    def write(self, data: StateRunInfo) -> None:
        self.videoWriter.write(data.image)
        self.paletteWriter.write(data.palette)

    def close(self):
        self.videoWriter.close()
        self.paletteWriter.close()


class State:
    @property
    @abstractmethod
    def is_recording(self) -> RecordingState:
        return RecordingState.IDLE

    def __init__(self, count: int):
        self.count = count

    @abstractmethod
    def run(self, info: StateRunInfo) -> 'State':
        pass


class IdleState(State):
    @property
    def is_recording(self) -> RecordingState:
        return RecordingState.IDLE

    def run(self, info: StateRunInfo) -> State:
        if info.is_kayaker_in_image:
            return RecordState(self.count, info.output_filename_template)
        else:
            return self


class RecordState(State):
    MAX_FRAME_COUNT_WITHOUT_KAYAKER = 50

    @property
    def is_recording(self) -> RecordingState:
        return RecordingState.RECORDING

    def __init__(self, count: int, output_filename_template: str):
        super().__init__(count)
        self.count += 1
        self.frames_without_kayaker_count = 0

        self.recorder: InfoRecorder = InfoRecorder(ImageWriter(output_filename_template.format(i=self.count)),
                                                   PerFrameNumpyWriter(output_filename_template.format(i=self.count)))

    def run(self, info: StateRunInfo):
        if self.frames_without_kayaker_count < self.MAX_FRAME_COUNT_WITHOUT_KAYAKER:
            if info.is_kayaker_in_image:
                self.frames_without_kayaker_count = 0
            else:
                self.frames_without_kayaker_count += 1
            self.recorder.write(info)
            return self
        else:
            self.recorder.close()
            return IdleState(self.count)


def chop(input_filenames: List[str],
         output_filename_template: str,
         left: int,
         top: int,
         right: int,
         bottom: int,
         video_writer: VideoWriter = None,
         palette_writer: DataWriter = None):
    """
    Args:
        :param video_writer: output writer for video data, defaults to ImageWriter
        :param palette_writer: output writer for palette data, defaults to PerFrameDataWriter
        :param input_filenames: file to open
        :param output_filename_template: file to write
        :param left: left border of ROI
        :param top: top border of ROI
        :param right: right border of ROI
        :param bottom: bottom border of ROI
    """

    for filename in input_filenames:
        cap = cv.VideoCapture(filename)

        score_hist = []
        state: State = IdleState(count=0)
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            original = np.copy(frame)

            roi_rgb = np.array(frame[top:bottom, left:right, :])

            scaled = cv.resize(roi_rgb, (int(roi_rgb.shape[1]/10), int(roi_rgb.shape[0]/10)),
                               interpolation=cv.INTER_NEAREST)

            z = np.float32(scaled.reshape(-1, 3))
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 5, 1.0)

            palette_shape = (5, 4, 3)
            k = palette_shape[0] * palette_shape[1]
            ret, label, center = cv.kmeans(z, k, None, criteria, 5,
                                           cv.KMEANS_RANDOM_CENTERS)
            palette = np.uint8(center)

            score = np.average(cv.cvtColor(np.reshape(palette, palette_shape), cv.COLOR_BGR2HSV)[:, :, 1])
            score_hist.append(score)
            if len(score_hist) > 50:
                score_hist.pop(0)
            avg_score = np.average(score_hist)

            print(avg_score)

            state = state.run(StateRunInfo(original, palette, avg_score > 20, output_filename_template))

            cv.rectangle(frame, (left, top), (right, bottom), state.is_recording.value, 3)
            if state.is_recording != RecordingState.IDLE:
                cv.putText(frame,
                           str(state.count),
                           (left, top - 10),
                           cv.FONT_HERSHEY_COMPLEX_SMALL,
                           2,
                           state.is_recording.value)

            cv.imshow('palette', cv.resize(np.reshape(palette, palette_shape), (500, 400),
                                           interpolation=cv.INTER_NEAREST))

            cv.imshow('original', cv.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2))))
            cv.imshow('roi', roi_rgb)

            if cv.waitKey(1) == 'q':
                break

        cap.release()
    cv.destroyAllWindows()

