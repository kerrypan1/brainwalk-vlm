"""
File Summary
------------
Defines the shared abstract interface for all model wrappers in this project.
Any model adapter under `models/` should inherit from `BaseVLM` so the
inference script can call each model with the same API.
"""
from abc import ABC, abstractmethod


class BaseVLM(ABC):
	"""Common contract for all VLM wrappers used by the pipeline."""

	name = "base"

	@abstractmethod
	def run(self, video_path, prompt):
		"""
		Run the model on one video clip and one prompt string.

		Expected return type across implementations:
		- `text_output`: generated text prediction
		- `embedding_or_none`: optional extra output (currently unused)
		"""
		raise NotImplementedError
