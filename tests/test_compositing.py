"""Compositing tests for use with pytest."""

import os

import numpy as np

import pytest

from moviepy import *


class ClipPixelTest:
    ALLOWABLE_COLOR_VARIATION = 3  # from 0-767: how much mismatch do we allow

    def __init__(self, clip):
        self.clip = clip

    def expect_color_at(self, ts, expected, xy=[0, 0]):
        frame = self.clip.frame_function(ts)
        r, g, b = expected
        actual = frame[xy[1]][xy[0]]
        diff = abs(actual[0] - r) + abs(actual[1] - g) + abs(actual[2] - b)

        mismatch = diff > ClipPixelTest.ALLOWABLE_COLOR_VARIATION
        assert (
            not mismatch
        ), "Expected (%02x,%02x,%02x) but got (%02x,%02x,%02x) at timestamp %s" % (
            *expected,
            *actual,
            ts,
        )


def test_clips_array(util):
    red = ColorClip((1024, 800), color=(255, 0, 0))
    green = ColorClip((1024, 800), color=(0, 255, 0))
    blue = ColorClip((1024, 800), color=(0, 0, 255))

    video = clips_array([[red, green, blue]])

    with pytest.raises(ValueError):  # duration not set
        video.with_effects([vfx.Resize(width=480)]).write_videofile(
            os.path.join(util.TMP_DIR, "test_clips_array.mp4")
        )


def test_clips_array_duration(util):
    filename = os.path.join(util.TMP_DIR, "test_clips_array.mp4")

    # NOTE: anyone knows what behaviour this sets ? If yes please replace
    # this comment.
    red = ColorClip((256, 200), color=(255, 0, 0))
    green = ColorClip((256, 200), color=(0, 255, 0))
    blue = ColorClip((256, 200), color=(0, 0, 255))

    video = clips_array([[red, green, blue]]).with_duration(5)
    with pytest.raises(AttributeError):  # fps not set
        video.write_videofile(filename)

    # this one should work correctly
    red.fps = green.fps = blue.fps = 30
    video = clips_array([[red, green, blue]]).with_duration(5)
    video.write_videofile(filename)


def test_concatenate_self(util):
    clip = BitmapClip([["AAA", "BBB"], ["CCC", "DDD"]], fps=1)
    target = BitmapClip([["AAA", "BBB"], ["CCC", "DDD"]], fps=1)

    concatenated = concatenate_videoclips([clip])

    concatenated.write_videofile(
        os.path.join(util.TMP_DIR, "test_concatenate_self.mp4")
    )
    assert concatenated == target


def test_concatenate_floating_point(util):
    """
    >>> print("{0:.20f}".format(1.12))
    1.12000000000000010658

    This test uses duration=1.12 to check that it still works when the clip
    duration is represented as being bigger than it actually is. Fixed in #1195.
    """
    clip = ColorClip([100, 50], color=[255, 128, 64], duration=1.12).with_fps(25.0)
    concat = concatenate_videoclips([clip])
    concat.write_videofile(os.path.join(util.TMP_DIR, "concat.mp4"), preset="ultrafast")


def test_blit_with_opacity():
    # has one second R, one second G, one second B
    size = (2, 2)
    clip1 = (
        ColorClip(size, color=(255, 0, 0), duration=1)
        + ColorClip(size, color=(0, 255, 0), duration=1)
        + ColorClip(size, color=(0, 0, 255), duration=1)
    )

    # overlay green at half opacity during first 2 sec
    clip2 = ColorClip(size, color=(0, 255, 0), duration=2).with_opacity(0.5)
    composite = CompositeVideoClip([clip1, clip2])
    bt = ClipPixelTest(composite)

    # red + 50% green
    bt.expect_color_at(0.5, (0x7F, 0x7F, 0x00))
    # green + 50% green
    bt.expect_color_at(1.5, (0x00, 0xFF, 0x00))
    # blue is after 2s, so keep untouched
    bt.expect_color_at(2.5, (0x00, 0x00, 0xFF))


def test_compositing_masks(util):
    # Has one R 30%, one G 30%, one B 30%
    clip1 = ColorClip((100, 100), (255, 0, 0, 76.5)).with_duration(2)
    clip2 = ColorClip((50, 50), (0, 255, 0, 76.5)).with_duration(2)
    clip3 = ColorClip((25, 25), (0, 0, 255, 76.5)).with_duration(2)

    compostite_clip1 = CompositeVideoClip(
        [clip1, clip2.with_position(("center", "center"))],
        bg_color=None,
    )
    compostite_clip2 = CompositeVideoClip(
        [compostite_clip1, clip3.with_position(("center", "center"))],
        bg_color=None,
    )

    # Load output file and check transparency
    frame = compostite_clip2.mask.get_frame(1)

    # We check opacity with one, two and three layers
    # Allow for a bit of tolerance (about 1%) to account
    # for rounding errors
    opacity1 = frame[50, 10]
    opacity2 = frame[50, 30]
    opacity3 = frame[50, 50]
    assert abs(opacity1 - 0.3) < 0.01
    assert abs(opacity2 - 0.51) < 0.01
    assert abs(opacity3 - 0.657) < 0.01


def test_compositing_with_transparency_colors(util):
    # Has one R 30%, one G 30%, one B 30%
    clip1 = ColorClip((100, 100), (255, 0, 0, 76.5)).with_duration(2)
    clip2 = ColorClip((50, 50), (0, 255, 0, 76.5)).with_duration(2)
    clip3 = ColorClip((25, 25), (0, 0, 255, 76.5)).with_duration(2)

    compostite_clip1 = CompositeVideoClip(
        [clip1, clip2.with_position(("center", "center"))],
        bg_color=None,  # No background color, so it should be transparent
    )
    compostite_clip2 = CompositeVideoClip(
        [compostite_clip1, clip3.with_position(("center", "center"))],
        bg_color=None,
    )

    # Load output file and check transparency
    frame = compostite_clip2.get_frame(1)
    mask = compostite_clip2.mask.get_frame(1)

    # We check color with 1 layer
    # We add a bit of tolerance (about 1%) to account
    # For possible rounding errors
    color1 = frame[50, 10]
    opacity1 = mask[50, 10]
    assert np.allclose(color1, [255, 0, 0], atol=1)
    assert abs(opacity1 - 0.3) < 0.01

    # With 2 layers
    color2 = frame[50, 30]
    opacity2 = mask[50, 30]
    assert np.allclose(color2, [105, 150, 0], atol=1)
    assert abs(opacity2 - 0.51) < 0.01

    # With 3 layers
    color3 = frame[50, 50]
    opacity3 = mask[50, 50]
    assert np.allclose(color3, [56, 81, 116], atol=1)
    assert abs(opacity3 - 0.657) < 0.01


def test_slide_in():
    duration = 0.1
    size = (10, 1)
    fps = 10
    color = (255, 0, 0)

    # left and right sides
    clip = ColorClip(
        color=color,
        duration=duration,
        size=size,
    ).with_fps(fps)

    for side in ["left", "right"]:
        new_clip = CompositeVideoClip(
            [clip.with_effects([vfx.SlideIn(duration, side)])]
        )

        for t in np.arange(0, duration, duration / fps):
            n_reds, n_reds_expected = (0, int(t * 100))

            if t:
                assert n_reds_expected

            if n_reds_expected == 7:  # skip 7 due to inaccurate frame
                continue

            for r, g, b in new_clip.get_frame(t)[0]:
                if r == color[0] and g == color[1] and g == color[2]:
                    n_reds += 1

            assert n_reds == n_reds_expected

    # top and bottom sides
    clip = ColorClip(
        color=color,
        duration=duration,
        size=(size[1], size[0]),
    ).with_fps(fps)

    for side in ["top", "bottom"]:
        new_clip = CompositeVideoClip(
            [clip.with_effects([vfx.SlideIn(duration, side)])]
        )
        for t in np.arange(0, duration, duration / fps):
            n_reds, n_reds_expected = (0, int(t * 100))

            if t:
                assert n_reds_expected

            if n_reds_expected == 7:  # skip 7 due to inaccurate frame
                continue

            for row in new_clip.get_frame(t):
                r, g, b = row[0]

                if r == color[0] and g == color[1] and g == color[2]:
                    n_reds += 1

            assert n_reds == n_reds_expected


def test_slide_out():
    duration = 0.1
    size = (11, 1)
    fps = 10
    color = (255, 0, 0)

    # left and right sides
    clip = ColorClip(
        color=color,
        duration=duration,
        size=size,
    ).with_fps(fps)

    for side in ["left", "right"]:
        new_clip = CompositeVideoClip(
            [clip.with_effects([vfx.SlideOut(duration, side)])]
        )

        for t in np.arange(0, duration, duration / fps):
            n_reds, n_reds_expected = (0, round(11 - t * 100, 6))

            if t:
                assert n_reds_expected

            for r, g, b in new_clip.get_frame(t)[0]:
                if r == color[0] and g == color[1] and g == color[2]:
                    n_reds += 1

            assert n_reds == n_reds_expected

    # top and bottom sides
    clip = ColorClip(
        color=color,
        duration=duration,
        size=(size[1], size[0]),
    ).with_fps(fps)

    for side in ["top", "bottom"]:
        new_clip = CompositeVideoClip(
            [clip.with_effects([vfx.SlideOut(duration, side)])]
        )
        for t in np.arange(0, duration, duration / fps):
            n_reds, n_reds_expected = (0, round(11 - t * 100, 6))

            if t:
                assert n_reds_expected

            for row in new_clip.get_frame(t):
                r, g, b = row[0]

                if r == color[0] and g == color[1] and g == color[2]:
                    n_reds += 1

            assert n_reds == n_reds_expected


def test_concatenate_with_masks(util):
    video_without_mask = (
        ColorClip(size=(10, 10), color=(255, 0, 0)).with_duration(1).with_fps(1)
    )
    video_with_mask = (
        ColorClip(size=(5, 5), color=(0, 255, 0))
        .with_duration(1)
        .with_fps(1)
        .with_mask()
    )

    output = os.path.join(util.TMP_DIR, "test_concatenate_with_masks.mp4")
    concatenate_videoclips([video_without_mask, video_with_mask]).write_videofile(
        output
    )


# ---------------------------------------------------------------------------
# Deterministic, reproducible frame compositing
# ---------------------------------------------------------------------------

# One distinct solid color per frame index, so the frame a clip is sampled on
# can be read straight from the pixel value.
_STEP_COLORS = {
    chr(65 + i): (10 * (i % 20) + 5, 10 * (i % 20) + 5, 10 * (i % 20) + 5)
    for i in range(26)
}


def _step_clip(fps, n_frames, start=0.0):
    """A 1x1 clip whose frame `i` has a unique, frame-index-encoded color."""
    frames = [[[chr(65 + i)]] for i in range(n_frames)]
    return BitmapClip(frames, fps=fps, color_dict=_STEP_COLORS).with_start(start)


def _step_color(index):
    return _STEP_COLORS[chr(65 + (index % 20))]


def test_composite_frame_snapped_to_clip_fps_grid():
    """A clip with a start offset is sampled on its own frame grid.

    At composite time ``t`` the clip must show its frame
    ``round((t - start) * fps)`` even when floating point drift makes the raw
    product land just below an integer (e.g. ``(0.7 - 0.2) * 10 == 4.999...``).
    """
    start, fps = 0.2, 10
    clip = _step_clip(fps, 10, start=start)
    composite = CompositeVideoClip([clip], size=(1, 1))

    # (t - start) * fps is 4.999999999999999 here: without snapping, int()
    # would select frame 4 instead of the expected frame 5.
    t = 0.7
    assert (t - start) * fps < 5  # the drift actually happens
    frame = composite.get_frame(t)
    assert tuple(int(v) for v in frame[0, 0]) == _step_color(5)


def test_composite_different_fps_first_frame_alignment():
    """Clips with different fps and start offsets each keep their own grid."""
    clip_10fps = _step_clip(10, 10, start=0.0)
    clip_7fps = _step_clip(7, 7, start=0.3).with_position((0, 0))
    composite = CompositeVideoClip(
        [clip_10fps, clip_7fps], size=(1, 1), bg_color=(0, 0, 0)
    )

    # Sample at composite times deliberately chosen to drift below integer
    # frame boundaries for at least one of the two clips.
    expectations = {
        0.30: (3, 0),  # (0.30-0.0)*10 -> frame 3, (0.30-0.3)*7 -> frame 0
        0.70: (7, 2),  # (0.70-0.0)*10 drifts to 6.999.., (0.70-0.3)*7 -> 2.8
        0.90: (9, 4),
    }
    for t, (idx_10, idx_7) in expectations.items():
        # Top clip (7 fps) wins because it is last in the same layer.
        frame = composite.get_frame(t)
        assert tuple(int(v) for v in frame[0, 0]) == _step_color(idx_7), t

        # Reading each clip through the composite's time mapping gives the
        # frame expected from its own grid.
        assert composite._clip_time(clip_10fps, t) == idx_10 / 10
        assert composite._clip_time(clip_7fps, t) == idx_7 / 7


def test_composite_same_pixels_independent_of_read_order():
    """Reading frames out of order must not change any returned frame.

    The sampled times include floating point drift boundaries, so this also
    pins each result to the frame dictated by the clip's own start/fps grid.
    """
    clip1 = _step_clip(10, 10, start=0.0)
    clip2 = _step_clip(7, 7, start=0.3).with_position((0, 0))
    composite = CompositeVideoClip([clip1, clip2], size=(1, 1), bg_color=(0, 0, 0))

    # t=0.7 -> (0.7-0.0)*10 drifts to 6.999.. and (0.7-0.3)*7 = 2.799..
    # t=0.9 -> (0.9-0.3)*7 drifts to 4.199.. while 0.05-grid times cover the rest
    times = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    forward = {t: composite.get_frame(t).copy() for t in times}
    shuffled = {t: composite.get_frame(t).copy() for t in reversed(times)}

    for t in times:
        np.testing.assert_array_equal(
            forward[t], shuffled[t], err_msg=f"frame differs at t={t}"
        )

    # Explicit expected frame of the top (7 fps, start 0.3) clip at each time.
    expected_top_frame = {
        0.0: None,  # clip2 not playing yet (starts at 0.3)
        0.1: None,
        0.3: 0,
        0.5: 1,
        0.7: 2,
        0.9: 4,
    }
    for t, idx in expected_top_frame.items():
        if idx is None:
            continue
        assert tuple(int(v) for v in forward[t][0, 0]) == _step_color(idx), t


def test_composite_mask_same_pixels_independent_of_read_order():
    """The memoized mask must always be the fully composited one."""
    clip1 = _step_clip(10, 10).with_position((0, 0)).with_mask()
    clip2 = _step_clip(10, 10).with_position((1, 0)).with_mask()
    composite = CompositeVideoClip([clip1, clip2], size=(2, 1), bg_color=None)

    times = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Reference masks computed standalone, before any RGB frame is requested.
    reference = CompositeVideoClip(
        [
            _step_clip(10, 10).with_position((0, 0)).with_mask(),
            _step_clip(10, 10).with_position((1, 0)).with_mask(),
        ],
        size=(2, 1),
        bg_color=None,
    )
    expected_masks = {t: reference.mask.get_frame(t).copy() for t in times}

    # Drive the RGB path. The memoized entry must always be the complete,
    # final mask rather than an intermediate mask captured mid-compositing.
    for t in times:
        composite.get_frame(t)
        np.testing.assert_allclose(composite.mask.precomputed[t], expected_masks[t])

    # Reading the masks back in a different order returns the same pixels.
    for t in reversed(times):
        np.testing.assert_allclose(
            composite.mask.get_frame(t),
            expected_masks[t],
            err_msg=f"mask differs at t={t}",
        )


def test_composite_memoized_mask_never_partial_during_blit():
    """No intermediate (partial) mask may ever be cached while blitting.

    The historical implementation wrote ``precomputed[t]`` after each clip,
    so the cache briefly held the mask with only the first clip applied. That
    partial value could be observed by another frame read and made the output
    depend on read order. This test spies on the cache through a wrapper around
    ``compose_on`` and fails as soon as a partial entry is stored.
    """
    clip1 = _step_clip(10, 10).with_position((0, 0)).with_mask()
    clip2 = _step_clip(10, 10).with_position((1, 0)).with_mask()
    composite = CompositeVideoClip([clip1, clip2], size=(2, 1), bg_color=None)
    t = 0.5

    full_mask = CompositeVideoClip(
        [
            _step_clip(10, 10).with_position((0, 0)).with_mask(),
            _step_clip(10, 10).with_position((1, 0)).with_mask(),
        ],
        size=(2, 1),
        bg_color=None,
    ).mask.get_frame(t)

    import inspect

    seen_cached_masks = []
    original_compose_on = clip2.compose_on
    accepts_frame_t = "frame_t" in inspect.signature(original_compose_on).parameters

    def spy(background, ct, background_mask, frame_t=None):
        if accepts_frame_t:
            result = original_compose_on(
                background, ct, background_mask, frame_t=frame_t
            )
        else:
            result = original_compose_on(background, ct, background_mask)
        # Record whatever is cached for this time after a blit. On a correct
        # implementation every such entry must already be the complete mask
        # (no partial, mid-compositing mask is ever cached).
        cached = composite.mask.precomputed.get(t)
        if cached is not None:
            seen_cached_masks.append(cached.copy())
        return result

    clip2.compose_on = spy
    composite.get_frame(t)

    # The final cache entry exists and is complete...
    assert t in composite.mask.precomputed
    np.testing.assert_allclose(composite.mask.precomputed[t], full_mask)
    # ...and nothing partial was ever observable during the blits.
    for cached in seen_cached_masks:
        np.testing.assert_allclose(cached, full_mask)


def test_composite_reproducible_across_runs():
    """Fresh composites built from the same clips yield identical frames.

    Times include floating point drift boundaries so reproducibility is
    asserted on the exact frame each clip's own start/fps grid prescribes.
    """

    def build():
        clip1 = _step_clip(10, 10, start=0.0).with_mask()
        clip2 = _step_clip(7, 7, start=0.3).with_position((0, 0)).with_mask()
        return CompositeVideoClip([clip1, clip2], size=(1, 1), bg_color=None)

    times = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    first = build()
    rgb_runs = [[build().get_frame(t).copy() for t in times] for _ in range(3)]
    mask_runs = [[build().mask.get_frame(t).copy() for t in times] for _ in range(3)]

    for run in rgb_runs[1:]:
        for i, t in enumerate(times):
            np.testing.assert_array_equal(
                rgb_runs[0][i], run[i], err_msg=f"rgb differs at t={t}"
            )
    for run in mask_runs[1:]:
        for i, t in enumerate(times):
            np.testing.assert_allclose(
                mask_runs[0][i], run[i], err_msg=f"mask differs at t={t}"
            )

    # Sanity check: the reference object matches every fresh run too and the
    # pixels correspond to the expected top-clip frame on its own grid.
    expected_top_frame = {0.3: 0, 0.5: 1, 0.7: 2, 0.9: 4}
    for i, t in enumerate(times):
        np.testing.assert_array_equal(first.get_frame(t), rgb_runs[0][i])
        if t in expected_top_frame:
            assert tuple(int(v) for v in rgb_runs[0][i][0, 0]) == _step_color(
                expected_top_frame[t]
            )


def test_composite_deterministic_with_memoize_mask_disabled():
    """Determinism must hold with mask memoization switched off as well."""
    times = [round(0.05 * i, 3) for i in range(20)]

    def build():
        clip1 = _step_clip(10, 10).with_mask()
        clip2 = _step_clip(7, 7).with_start(0.3).with_position((0, 0))
        clip2 = clip2.with_mask()
        return CompositeVideoClip(
            [clip1, clip2],
            size=(1, 1),
            bg_color=None,
            memoize_mask=False,
        )

    composite = build()
    forward = {t: composite.get_frame(t).copy() for t in times}
    shuffled = {t: composite.get_frame(t).copy() for t in reversed(times)}
    for t in times:
        np.testing.assert_array_equal(forward[t], shuffled[t])
        assert not hasattr(composite.mask, "precomputed")


def test_composite_use_bgclip_snapped_to_background_grid():
    """The background clip is sampled on its own start/fps grid too."""
    # Background at 6 fps; foreground is fully opaque and only 0.4 s long so
    # it has ended at the sampled time and only the background is visible.
    background = _step_clip(6, 6).with_start(0.2)
    foreground = _step_clip(4, 2).with_start(0.2)  # ends at 0.2 + 0.5 = 0.7
    composite = CompositeVideoClip(
        [background, foreground], size=(1, 1), use_bgclip=True
    )

    # (0.7 - 0.2) * 6 drifts to 2.9999... just below the exact frame boundary.
    t = 0.7
    raw = (t - 0.2) * 6
    assert int(raw) != int(round(raw))  # floating point drift is present
    expected_index = int(round(raw))  # frame 3
    # The background resolves to its own grid frame.
    np.testing.assert_array_equal(
        composite.bg.get_frame(composite._clip_time(composite.bg, t)),
        background.get_frame(expected_index / 6.0),
    )
    frame = composite.get_frame(t)
    assert tuple(int(v) for v in frame[0, 0]) == _step_color(expected_index)

    # Independent rebuilds give identical pixels.
    rebuilt = CompositeVideoClip(
        [
            _step_clip(6, 6).with_start(0.2),
            _step_clip(4, 2).with_start(0.2),
        ],
        size=(1, 1),
        use_bgclip=True,
    )
    np.testing.assert_array_equal(rebuilt.get_frame(t), frame)


def test_composite_background_mask_snapped_to_its_own_grid():
    """A masked background keeps RGB and mask on the same snapped grid."""
    background = _step_clip(10, 10).with_mask()
    foreground = _step_clip(7, 7, start=0.3)
    composite = CompositeVideoClip([background, foreground], size=(1, 1), bg_color=None)

    t = 0.7  # (0.7 - 0.0) * 10 drifts to 6.999..
    rgb = composite.get_frame(t).copy()
    mask = composite.mask.get_frame(t).copy()

    rebuilt = CompositeVideoClip(
        [_step_clip(10, 10).with_mask(), _step_clip(7, 7, start=0.3)],
        size=(1, 1),
        bg_color=None,
    )
    np.testing.assert_array_equal(rebuilt.get_frame(t), rgb)
    np.testing.assert_allclose(rebuilt.mask.get_frame(t), mask)


if __name__ == "__main__":
    pytest.main()
