"""Resolved latent and layout geometry shared by codecs, routing, and models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GeometryContract:
    """Physical geometry resolved from the selected codec configuration.

    The paper recipe remains the defaults. Extensions may choose another
    vision grid, token budget, or text latent width as long as every stage
    consumes the same resolved contract.
    """

    vision_tokens: int = 256
    vision_latent_dim: int = 768
    text_latent_dim: int = 512
    vision_grid_size: tuple[int, int] = (16, 16)

    def validate(self) -> "GeometryContract":
        for name, value in (
            ("vision_tokens", self.vision_tokens),
            ("vision_latent_dim", self.vision_latent_dim),
            ("text_latent_dim", self.text_latent_dim),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.vision_grid_size) is not tuple
            or len(self.vision_grid_size) != 2
            or any(type(value) is not int or value <= 0 for value in self.vision_grid_size)
        ):
            raise ValueError("vision_grid_size must be a pair of positive integers")
        if self.vision_tokens != self.vision_grid_size[0] * self.vision_grid_size[1]:
            raise ValueError(
                "vision_tokens must equal vision_grid_size[0] * vision_grid_size[1]"
            )
        return self

    @property
    def vision_prefix_tokens(self) -> int:
        return 8

    @property
    def text_prefix_tokens(self) -> int:
        return 12

    @property
    def vision_layout_tokens(self) -> int:
        return self.vision_prefix_tokens + self.vision_tokens

    @classmethod
    def from_config(cls, config: object) -> "GeometryContract":
        vision = config.codecs.vision
        text = config.codecs.text
        contract = cls(
            vision_tokens=int(vision.latent_tokens),
            vision_latent_dim=int(vision.latent_dim),
            text_latent_dim=int(text.latent_dim),
            vision_grid_size=tuple(int(value) for value in vision.grid_size),
        )
        return contract.validate()
