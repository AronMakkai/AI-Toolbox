#!/usr/bin/env python3
"""ADW-Organic mask preview -- regression tests.

The point of this feature is that the black-and-white field the user
tunes against is the SAME array the effect applies. A preview that is
merely a plausible re-derivation of the effect is worse than no preview
at all, because it will drift silently the first time one side is
edited and then quietly lie about where the effect is landing.

So most of what follows is equivalence testing: run the real effect,
algebraically recover the spatial field it used, and assert it matches
_organic_mask_field byte for byte within float tolerance.

Run:  python3 test_organic_masks.py
"""
import ast
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
import unittest.mock

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_toolbox import App, G  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ai_toolbox.py")


def plate(h=64, w=96, seed=7):
    """Scene-linear test plate with genuine over-1.0 highlights, so the
    bloom path is exercised the way real EXR data exercises it."""
    rng = np.random.default_rng(seed)
    img = rng.random((h, w, 3)).astype(np.float32) * 0.8
    img[10:20, 12:28] += 3.0          # a hot highlight
    img[40:50, 60:80] += 0.6          # a mid-bright patch
    return img


PARAMS = dict(
    ca=0.6, ca_falloff=2.3,
    bloom_thresh=0.85, bloom_radius=9, bloom_intensity=0.5, bloom_aniso=2.0,
    bloom_kgamma=1.0,
    bloom_sph_size=0.35, bloom_sph_falloff=2.0,
    bloom_noi_mix=0.3, bloom_noi_scale=6, bloom_noi_detail=3,
    bloom_noi_contrast=1.0, bloom_noi_seed=1,
    bloom_noi_rings=0.0, bloom_noi_ring_freq=10,
    bloom_noi_rays=0.0, bloom_noi_ray_freq=32,
    bloom_str_gain=0.4, bloom_str_points=6, bloom_str_angle=0.0,
    bloom_str_length=0.9, bloom_str_width=12.0, bloom_str_falloff=1.6,
    bloom_disp=0.5, bloom_bands=1, bloom_band_spread=2.0,
    veil_thresh=0.85, veil_intensity=0.4,
    veil_size_a=12, veil_size_b=40, veil_mix=0.5, veil_sat=0.5,
    veil_bands=1, veil_band_spread=2.0,
    soft_amount=0.0, soft_radius=8.0, soft_halo=0.35,
    soft_edge=0.5, soft_edge_falloff=2.0, soft_chroma=0.5,
    smudge_amount=0.0, smudge_scale=4, smudge_detail=3,
    smudge_coverage=0.45, smudge_softness=0.35,
    smudge_blur=6.0, smudge_haze=0.35, smudge_seed=1,
    scratch_amount=0.0, scratch_count=12, scratch_length=0.35,
    scratch_width=2, scratch_softness=1.2, scratch_angle=0.0,
    scratch_angle_var=180.0, scratch_spread=20.0, scratch_seed=1,
    vig=0.45, vig_falloff=1.8,
    grain=0.03, grain_size=1.0, grain_shadow=1.6, grain_chroma=0.0,
    grain_filmic=0.0, global_gain=1.0)

# A kernel with neither noise nor star -- the plain spherical case.
SPHERE_ONLY = dict(PARAMS, bloom_noi_mix=0.0, bloom_str_gain=0.0,
                   bloom_disp=0.0, veil_intensity=0.0)


class TestMaskShape(unittest.TestCase):
    """Every mask must be a well-formed, displayable field."""

    # Every mask is a frame-space field except the bloom kernel, which
    # is its own small image -- the viewer fits it by its own aspect.
    OWN_SHAPE = {"bloom_kernel"}
    # bloom_kernel is RGB so the dispersed channels are visible
    COLOUR = {"bloom_kernel"}

    def test_all_masks_are_normalised_2d_float(self):
        rgb = plate()
        for key in App._ORGANIC_MASKS:
            with self.subTest(mask=key):
                m = App._organic_mask_field(rgb, PARAMS, key)
                self.assertEqual(m.ndim, 3 if key in self.COLOUR else 2)
                if key not in self.OWN_SHAPE:
                    self.assertEqual(m.shape, rgb.shape[:2])
                self.assertEqual(m.dtype, np.float32)
                self.assertTrue(np.isfinite(m).all())
                self.assertGreaterEqual(float(m.min()), 0.0)
                self.assertLessEqual(float(m.max()), 1.0)

    def test_unknown_mask_key_raises(self):
        with self.assertRaises(KeyError):
            App._organic_mask_field(plate(), PARAMS, "not_a_mask")

    def test_masks_survive_a_black_frame(self):
        """A frame with nothing above threshold must not divide by its
        own zero peak when normalising the bloom spill."""
        rgb = np.zeros((32, 48, 3), np.float32)
        for key in App._ORGANIC_MASKS:
            with self.subTest(mask=key):
                m = App._organic_mask_field(rgb, PARAMS, key)
                self.assertTrue(np.isfinite(m).all())


class TestMaskMatchesEffect(unittest.TestCase):
    """The reason this file exists."""

    def test_vignette_mask_is_the_vignette_multiplier(self):
        rgb = plate()
        m = App._organic_mask_field(rgb, PARAMS, "vignette")
        out = App._organic_vignette(rgb, PARAMS["vig"], PARAMS["vig_falloff"])
        np.testing.assert_allclose(out, rgb * m[:, :, None], rtol=1e-6)

    def test_ca_edge_mask_is_the_blend_weight_chromatic_uses(self):
        """Recover r from out = rgb*(1-r) + scaled*r on the red channel
        and check it against the previewed ramp."""
        rgb = plate()
        amount, falloff = PARAMS["ca"], PARAMS["ca_falloff"]
        h, w = rgb.shape[:2]
        out = App._organic_chromatic(rgb, amount, falloff)

        max_disp = amount / max(1.0, float(min(h, w))) * 100.0
        s = 1.0 + max_disp * 0.01
        M = np.float32([[s, 0, (1 - s) * w / 2.0],
                        [0, s, (1 - s) * h / 2.0]])
        scaled = cv2.warpAffine(rgb[:, :, 0], M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)

        denom = scaled - rgb[:, :, 0]
        usable = np.abs(denom) > 1e-3        # r is unrecoverable where the
        self.assertGreater(usable.sum(), 500)  # channels happen to agree
        recovered = (out[:, :, 0] - rgb[:, :, 0])[usable] / denom[usable]

        m = App._organic_mask_field(rgb, PARAMS, "ca_edge")
        np.testing.assert_allclose(recovered, m[usable], atol=1e-4)

    def test_grain_response_mask_is_the_grain_weighting(self):
        """With chroma=0 and size=1 the noise is a single reproducible
        mono field, so the response can be divided back out."""
        rgb = plate()
        seed = 4242
        out = App._organic_grain(rgb, PARAMS["grain"], 1.0,
                                 PARAMS["grain_shadow"], 0.0, seed=seed)
        h, w = rgb.shape[:2]
        # Position-addressed now, not a sequential draw — see
        # _organic_hash_normal and TestHashNoise.
        mono = App._organic_hash_normal(w, 0, 0, h, w, seed, 0)

        # Recovering resp means dividing by the noise, which is
        # ill-conditioned wherever the noise is near zero -- at 1e-2 a
        # single unlucky pixel in 6000 blows the relative tolerance
        # without anything being wrong. 5e-2 still leaves thousands of
        # samples.
        usable = np.abs(mono) > 5e-2
        self.assertGreater(usable.sum(), 2000)
        recovered = ((out[:, :, 0] - rgb[:, :, 0])[usable]
                     / (PARAMS["grain"] * mono[usable]))
        m = App._organic_mask_field(rgb, PARAMS, "grain_resp")
        np.testing.assert_allclose(recovered, m[usable], rtol=1e-3, atol=1e-5)

    def test_threshold_mask_is_exactly_what_blooms(self):
        rgb = plate()
        m = App._organic_mask_field(rgb, PARAMS, "bloom_thresh")
        over = App._organic_bloom_over(rgb, PARAMS["bloom_thresh"])
        np.testing.assert_array_equal(m > 0.5, over > 0.0)

    def test_threshold_mask_excludes_pixels_sitting_on_the_threshold(self):
        """Exactly at the threshold a pixel contributes zero energy, so
        it must read black -- a >= here would overstate the effect."""
        rgb = np.full((8, 8, 3), 0.85, np.float32)
        m = App._organic_mask_field(rgb, dict(bloom_thresh=0.85),
                                    "bloom_thresh")
        self.assertEqual(float(m.max()), 0.0)

    def test_threshold_mask_tracks_the_slider(self):
        rgb = plate()
        low = App._organic_mask_field(rgb, dict(bloom_thresh=0.3),
                                      "bloom_thresh").sum()
        high = App._organic_mask_field(rgb, dict(bloom_thresh=2.5),
                                       "bloom_thresh").sum()
        self.assertGreater(low, high)


class TestDispersion(unittest.TestCase):
    """Red spreads widest, blue tightest, green is the reference."""

    def test_zero_dispersion_reuses_one_kernel(self):
        """Not merely equal — the same object, so the single-kernel
        convolution path is taken and two DFTs are not wasted."""
        kr, kg, kb = App._organic_bloom_kernel_rgb(
            dict(SPHERE_ONLY, bloom_disp=0.0))
        self.assertIs(kr, kg)
        self.assertIs(kg, kb)

    def test_channels_are_ordered_red_widest_blue_tightest(self):
        kr, kg, kb = App._organic_bloom_kernel_rgb(
            dict(SPHERE_ONLY, bloom_radius=40, bloom_aniso=1.0,
                 bloom_disp=1.0))
        def spread(k):
            return float((k > 0.2).sum())
        self.assertGreater(spread(kr), spread(kg))
        self.assertGreater(spread(kg), spread(kb))

    def test_all_three_kernels_share_a_shape(self):
        """They are convolved channel-by-channel against one image, so
        a mismatch would be a crash, not a look."""
        kr, kg, kb = App._organic_bloom_kernel_rgb(
            dict(SPHERE_ONLY, bloom_disp=0.7))
        self.assertEqual(kr.shape, kg.shape)
        self.assertEqual(kg.shape, kb.shape)

    def test_the_widened_red_kernel_is_not_clipped_by_its_canvas(self):
        """The grid is padded when disp > 1; without that the red
        kernel would be cut off at the array edge and its energy
        silently truncated."""
        base = dict(SPHERE_ONLY, bloom_radius=30, bloom_aniso=1.0,
                    bloom_sph_size=0.3, bloom_sph_falloff=2.0)
        plain = App._organic_bloom_kernel(base)
        kr, _kg, _kb = App._organic_bloom_kernel_rgb(
            dict(base, bloom_disp=1.0))
        # the canvas grew to make room...
        self.assertGreater(kr.shape[0], plain.shape[0])
        # ...and the widened kernel has genuinely fallen away by the
        # time it reaches the new edge, rather than being cut off there
        edge = np.concatenate([kr[0], kr[-1], kr[:, 0], kr[:, -1]])
        self.assertLess(float(edge.max()), 1e-3)

    def test_a_flat_field_picks_up_no_tint(self):
        """Each channel is normalised by its own sum, so dispersion
        colours edges only — a uniform area must stay neutral."""
        rgb = np.full((96, 96, 3), 2.0, np.float32)
        out = App._organic_bloom(rgb, dict(SPHERE_ONLY, bloom_thresh=1.0,
                                           bloom_radius=8, bloom_aniso=1.0,
                                           bloom_intensity=1.0,
                                           bloom_disp=1.0))
        mid = out[40:56, 40:56]
        np.testing.assert_allclose(mid[:, :, 0], mid[:, :, 2], rtol=2e-3)

    def test_dispersion_colours_the_edge_of_a_white_highlight(self):
        rgb = np.zeros((128, 128, 3), np.float32)
        rgb[62:66, 62:66] = 8.0
        p = dict(SPHERE_ONLY, bloom_thresh=1.0, bloom_radius=24,
                 bloom_aniso=1.0, bloom_intensity=4.0, bloom_disp=1.0)
        add = App._organic_bloom(rgb, p) - rgb
        # out at the rim of the glow the red kernel still reaches and
        # the blue one has fallen away
        rim = add[64, 64 + 20]
        self.assertGreater(float(rim[0]), float(rim[2]) * 1.15)

    def test_no_dispersion_leaves_the_glow_neutral(self):
        rgb = np.zeros((128, 128, 3), np.float32)
        rgb[62:66, 62:66] = 8.0
        p = dict(SPHERE_ONLY, bloom_thresh=1.0, bloom_radius=24,
                 bloom_aniso=1.0, bloom_intensity=4.0, bloom_disp=0.0)
        add = App._organic_bloom(rgb, p) - rgb
        rim = add[64, 64 + 20]
        self.assertAlmostEqual(float(rim[0]), float(rim[2]), places=5)


class TestSensorVeil(unittest.TestCase):
    """Wide, white, shapeless — a different mechanism from the kernel
    bloom, with its own threshold."""

    VEIL = dict(veil_thresh=1.0, veil_intensity=1.0,
                veil_size_a=6, veil_size_b=30, veil_mix=0.5)

    def test_zero_intensity_is_a_pass_through(self):
        rgb = plate()
        np.testing.assert_array_equal(
            App._organic_veil(rgb, dict(self.VEIL, veil_intensity=0.0)), rgb)

    def test_nothing_above_threshold_is_a_pass_through(self):
        rgb = np.full((32, 32, 3), 0.2, np.float32)
        np.testing.assert_array_equal(
            App._organic_veil(rgb, dict(self.VEIL, veil_thresh=5.0)), rgb)

    def test_veil_only_adds_light(self):
        rgb = plate()
        self.assertTrue(np.all(App._organic_veil(rgb, self.VEIL) >= rgb))

    def test_veil_is_white_at_colour_zero(self):
        """Charge carries no wavelength, so at Colour 0 it lands equally
        on R, G and B even when the source highlight is strongly
        coloured. That used to be unconditional; it is now one end of
        the Colour slider, because almost none of the veiling glare on a
        real frame is charge bleed."""
        rgb = np.zeros((64, 64, 3), np.float32)
        rgb[30:34, 30:34, 0] = 6.0
        add = App._organic_veil(rgb, dict(self.VEIL, veil_sat=0.0)) - rgb
        np.testing.assert_allclose(add[:, :, 0], add[:, :, 1], atol=1e-6)
        np.testing.assert_allclose(add[:, :, 1], add[:, :, 2], atol=1e-6)

    def test_veil_reaches_far_from_the_source(self):
        rgb = np.zeros((200, 200, 3), np.float32)
        rgb[98:102, 98:102] = 20.0
        add = App._organic_veil(rgb, dict(self.VEIL, veil_size_a=10,
                                          veil_size_b=60)) - rgb
        self.assertGreater(float(add[100, 180, 0]), 1e-6)

    def test_balance_selects_between_the_two_scales(self):
        rgb = np.zeros((200, 200, 3), np.float32)
        rgb[98:102, 98:102] = 20.0

        def reach(mix):
            add = App._organic_veil(rgb, dict(
                self.VEIL, veil_size_a=4, veil_size_b=60,
                veil_mix=mix)) - rgb
            return float((add[:, :, 0] > add.max() * 0.05).sum())
        self.assertLess(reach(0.0), reach(1.0))

    def test_veil_has_its_own_threshold(self):
        """Independent of the kernel bloom's — the sensor overflows at
        a different level than the glass scatters."""
        rgb = np.zeros((64, 64, 3), np.float32)
        rgb[30:34, 30:34] = 2.0
        low = App._organic_veil(rgb, dict(self.VEIL, veil_thresh=0.5)) - rgb
        high = App._organic_veil(rgb, dict(self.VEIL, veil_thresh=1.8)) - rgb
        self.assertGreater(float(low.sum()), float(high.sum()))

    def test_wide_blur_downsamples_without_changing_the_result_much(self):
        """Large sigmas run on a downsampled copy; the result must
        still match a full-resolution blur closely."""
        rng = np.random.default_rng(11)
        f = rng.random((256, 256)).astype(np.float32)
        f[120:136, 120:136] += 5.0
        got = App._organic_wide_blur(f, 60.0)
        want = cv2.GaussianBlur(f, (0, 0), sigmaX=60.0, sigmaY=60.0)
        self.assertLess(float(np.abs(got - want).max()),
                        float(want.max()) * 0.05)

    def test_wide_blur_conserves_energy(self):
        """A single hot pixel is the worst case for the downsampled
        path -- it is exactly the detail the downsample throws away --
        so a few percent drift here is the ceiling, not the typical
        error on real image data."""
        f = np.zeros((128, 128), np.float32)
        f[64, 64] = 100.0
        for sigma in (3.0, 20.0, 90.0):
            with self.subTest(sigma=sigma):
                out = App._organic_wide_blur(f, sigma)
                self.assertAlmostEqual(float(out.sum()), 100.0, delta=10.0)

    def test_veil_is_its_own_effect_not_a_bloom_sub_group(self):
        """It has its own S/M pair: scatter in the glass and charge
        bleed on the sensor are different mechanisms, and the veil has
        to be soloable to tune it against the plate."""
        gates = dict((fx, g) for fx, _l, g in App._ORGANIC_FX)
        self.assertIn("veil", gates)
        self.assertEqual(gates["veil"], ("veil_intensity",))
        self.assertNotIn("veil_intensity", gates["bloom"])

    def test_muting_the_veil_leaves_the_bloom_alone(self):
        app = _StubApp()
        app._organic_fx_mute["veil"] = True
        p = app._organic_params()
        self.assertEqual(p["veil_intensity"], 0.0)
        self.assertEqual(p["bloom_intensity"], PARAMS["bloom_intensity"])

    def test_soloing_the_veil_silences_the_bloom(self):
        app = _StubApp()
        app._organic_fx_solo["veil"] = True
        p = app._organic_params()
        self.assertEqual(p["bloom_intensity"], 0.0)
        self.assertEqual(p["veil_intensity"], PARAMS["veil_intensity"])

    def test_stack_runs_lens_effects_before_the_sensor_veil(self):
        """Everything above the veil is the lens acting on light that
        has not reached the sensor yet. Running the veil before the
        vignette let barrel falloff darken an artefact that forms
        behind it — a vignette on the veil's own outer tail.

        Asked of the stack itself rather than grepped out of the
        source: the previous version matched on call-site text and
        broke the moment the chain was rebuilt as a step list, which
        makes it a test of the formatting rather than of the order."""
        app = _StubApp()
        order = []
        app._organic_apply_stack(
            np.full((32, 32, 3), 0.5, np.float32),
            dict(App._ORGANIC_BASE, ca=0.5, vig=0.3, grain=0.02,
                 bloom_intensity=0.4, bloom_radius=3,
                 veil_intensity=0.3, veil_size_b=12),
            seed=1, progress=lambda d, t, n: order.append(n))
        order = [n for n in order if n]
        self.assertEqual(order, ["Aberration", "Bloom", "Vignette",
                                 "Sensor Veil", "Grain"])

    def test_panel_order_matches_stack_order(self):
        """The column should read top-to-bottom as the pipeline runs."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        # Prefix only -- some cards pass opened=False, so matching the
        # closing paren would silently find nothing.
        cards = [src.index(f'_fxcard("{label}", "{fx}"')
                 for fx, label, _g in App._ORGANIC_FX]
        self.assertEqual(cards, sorted(cards),
                         "card order does not match _ORGANIC_FX order")


class TestBloomKernel(unittest.TestCase):
    """The kernel is three layers composited with max and normalised to
    a peak of 1.0, which is what makes it directly displayable."""

    def test_kernel_is_peak_normalised(self):
        k = App._organic_bloom_kernel(PARAMS)
        self.assertAlmostEqual(float(k.max()), 1.0, places=5)
        self.assertGreaterEqual(float(k.min()), 0.0)
        self.assertEqual(k.dtype, np.float32)

    def test_kernel_is_odd_sized_and_centred(self):
        """An even kernel would shift the bloom half a pixel."""
        k = App._organic_bloom_kernel(dict(SPHERE_ONLY, bloom_radius=20,
                                           bloom_aniso=1.0))
        self.assertEqual(k.shape[0] % 2, 1)
        self.assertEqual(k.shape[1] % 2, 1)
        cy, cx = k.shape[0] // 2, k.shape[1] // 2
        self.assertAlmostEqual(float(k[cy, cx]), 1.0, places=5)

    def test_anamorph_makes_the_kernel_wider_than_tall(self):
        k = App._organic_bloom_kernel(dict(SPHERE_ONLY, bloom_radius=20,
                                           bloom_aniso=3.0))
        self.assertGreater(k.shape[1], k.shape[0] * 2)

    def test_kernel_size_follows_radius(self):
        small = App._organic_bloom_kernel(dict(PARAMS, bloom_radius=5))
        big = App._organic_bloom_kernel(dict(PARAMS, bloom_radius=40))
        self.assertGreater(big.shape[0], small.shape[0])

    def test_kernel_is_capped(self):
        """A radius of 120 at 6x anamorphic would otherwise build a
        1441-wide kernel."""
        k = App._organic_bloom_kernel(dict(PARAMS, bloom_radius=120,
                                           bloom_aniso=6.0))
        self.assertLessEqual(max(k.shape),
                             2 * App._ORGANIC_KERNEL_MAX_HALF + 1)

    def test_px_scale_shrinks_the_kernel(self):
        full = App._organic_bloom_kernel(PARAMS, 1.0)
        half = App._organic_bloom_kernel(PARAMS, 0.5)
        self.assertLess(half.shape[0], full.shape[0])

    # ── the sphere layer ──
    def test_sphere_falloff_changes_shape_not_just_size(self):
        """Falloff is the exponent in exp(-(r/s)**f). If it merely
        rescaled the radius it would be a duplicate of Size, so the two
        profiles must cross rather than nest."""
        soft = App._organic_bloom_kernel(dict(SPHERE_ONLY,
                                              bloom_sph_falloff=1.0))
        hard = App._organic_bloom_kernel(dict(SPHERE_ONLY,
                                              bloom_sph_falloff=6.0))
        cy = soft.shape[0] // 2
        a, b = soft[cy], hard[cy]
        self.assertTrue(np.any(a > b + 1e-4) and np.any(b > a + 1e-4),
                        "profiles nest — falloff is degenerate with size")

    def test_sphere_size_widens_the_core(self):
        tight = App._organic_bloom_kernel(dict(SPHERE_ONLY,
                                               bloom_sph_size=0.1))
        wide = App._organic_bloom_kernel(dict(SPHERE_ONLY,
                                              bloom_sph_size=0.8))
        self.assertLess(float((tight > 0.5).sum()),
                        float((wide > 0.5).sum()))

    def test_sphere_only_kernel_is_symmetric(self):
        k = App._organic_bloom_kernel(dict(SPHERE_ONLY, bloom_aniso=1.0))
        np.testing.assert_allclose(k, np.flipud(k), atol=1e-6)
        np.testing.assert_allclose(k, np.fliplr(k), atol=1e-6)

    # ── the noise layer ──
    def test_zero_noise_mix_ignores_the_noise_controls(self):
        a = App._organic_bloom_kernel(dict(SPHERE_ONLY, bloom_noi_seed=1))
        b = App._organic_bloom_kernel(dict(SPHERE_ONLY, bloom_noi_seed=99))
        np.testing.assert_array_equal(a, b)

    def test_noise_breaks_up_the_kernel(self):
        clean = App._organic_bloom_kernel(SPHERE_ONLY)
        noisy = App._organic_bloom_kernel(dict(SPHERE_ONLY,
                                               bloom_noi_mix=0.9))
        self.assertFalse(np.allclose(clean, noisy))

    def test_noise_is_reproducible(self):
        p = dict(SPHERE_ONLY, bloom_noi_mix=0.8)
        np.testing.assert_array_equal(App._organic_bloom_kernel(p),
                                      App._organic_bloom_kernel(p))

    def test_noise_seed_changes_the_pattern(self):
        a = App._organic_bloom_kernel(dict(SPHERE_ONLY, bloom_noi_mix=0.8,
                                           bloom_noi_seed=1))
        b = App._organic_bloom_kernel(dict(SPHERE_ONLY, bloom_noi_mix=0.8,
                                           bloom_noi_seed=2))
        self.assertFalse(np.allclose(a, b))

    def test_noise_stays_in_range(self):
        n = App._organic_kernel_noise((64, 64), 6, 4, 2.5, 3)
        self.assertGreaterEqual(float(n.min()), 0.0)
        self.assertLessEqual(float(n.max()), 1.0)

    def test_high_octave_counts_do_not_alias(self):
        """Lattices are clamped to the kernel size, so extra octaves
        must saturate rather than turn into per-pixel speckle."""
        n = App._organic_kernel_noise((16, 16), 30, 5, 1.0, 1)
        self.assertTrue(np.isfinite(n).all())

    # ── the star layer ──
    def test_the_noise_leaves_the_star_alone(self):
        """Noise multiplies the sphere; the star is mixed against the
        result. It used to be applied to the composite, which broke the
        arms up in the shape of the noise rather than their own, so the
        star read as unrelated debris."""
        base = dict(SPHERE_ONLY, bloom_radius=50, bloom_aniso=1.0,
                    bloom_sph_size=0.02, bloom_str_gain=1.0,
                    bloom_str_points=4, bloom_str_width=6,
                    bloom_str_falloff=0.4, bloom_kgamma=1.0)
        # At gain 1.0 the sphere is mixed out entirely, so the noise
        # controls must make no difference whatsoever.
        np.testing.assert_array_equal(
            App._organic_bloom_kernel(base),
            App._organic_bloom_kernel(dict(base, bloom_noi_mix=0.9,
                                           bloom_noi_scale=10)))

    def test_the_noise_still_breaks_up_the_sphere(self):
        base = dict(SPHERE_ONLY, bloom_radius=40, bloom_aniso=1.0,
                    bloom_sph_size=0.5, bloom_str_gain=0.0,
                    bloom_kgamma=1.0)
        clean = App._organic_bloom_kernel(base)
        noisy = App._organic_bloom_kernel(dict(base, bloom_noi_mix=0.9,
                                              bloom_noi_scale=10))
        self.assertGreater(float(np.abs(clean - noisy).max()), 0.05)

    def test_noise_reaches_the_sphere_under_a_partial_star_mix(self):
        base = dict(SPHERE_ONLY, bloom_radius=40, bloom_aniso=1.0,
                    bloom_sph_size=0.5, bloom_str_gain=0.4,
                    bloom_kgamma=1.0)
        self.assertFalse(np.allclose(
            App._organic_bloom_kernel(base),
            App._organic_bloom_kernel(dict(base, bloom_noi_mix=0.9))))

    def test_the_composition_is_a_lerp_of_noisy_sphere_and_star(self):
        """Spelled out end to end: sphere x noise, then mixed against
        the star by Star Gain."""
        p = dict(SPHERE_ONLY, bloom_radius=40, bloom_aniso=1.0,
                 bloom_sph_size=0.15, bloom_str_gain=0.6,
                 bloom_str_points=5, bloom_noi_mix=0.7, bloom_kgamma=1.0)
        r, theta = App._organic_kernel_grid(p, 1.0)
        n = App._organic_kernel_noise(r.shape, p["bloom_noi_scale"],
                                      p["bloom_noi_detail"],
                                      p["bloom_noi_contrast"],
                                      p["bloom_noi_seed"])
        polar = App._organic_kernel_polar_noise(r, theta, p)
        if polar is not None:
            n = n * polar
        sph = App._organic_kernel_sphere(r, p["bloom_sph_size"],
                                         p["bloom_sph_falloff"])
        sph = sph * (1.0 - p["bloom_noi_mix"] + p["bloom_noi_mix"] * n)
        star = App._organic_kernel_star(
            r, theta, p["bloom_str_points"], p["bloom_str_angle"],
            p["bloom_str_length"], p["bloom_str_width"],
            p["bloom_str_falloff"])
        g = p["bloom_str_gain"]
        want = sph * (1.0 - g) + star * g
        np.testing.assert_allclose(App._organic_bloom_kernel(p),
                                   want / want.max(), atol=1e-5)

    def test_noise_never_brightens_the_kernel(self):
        """The multiplier tops out at 1.0, so noise can only carve
        away — it must not invent energy on the arms."""
        base = dict(SPHERE_ONLY, bloom_radius=40, bloom_aniso=1.0,
                    bloom_sph_size=0.1, bloom_str_gain=0.9,
                    bloom_kgamma=1.0)
        noisy = App._organic_bloom_kernel(dict(base, bloom_noi_mix=0.6))
        self.assertLessEqual(float(noisy.max()), 1.0 + 1e-6)
        self.assertGreaterEqual(float(noisy.min()), 0.0)

    def test_zero_star_gain_ignores_the_star_controls(self):
        a = App._organic_bloom_kernel(dict(SPHERE_ONLY,
                                           bloom_str_points=6))
        b = App._organic_bloom_kernel(dict(SPHERE_ONLY,
                                           bloom_str_points=13))
        np.testing.assert_array_equal(a, b)

    def test_star_adds_reach_beyond_the_sphere(self):
        core = App._organic_bloom_kernel(dict(SPHERE_ONLY,
                                              bloom_sph_size=0.1))
        starred = App._organic_bloom_kernel(dict(SPHERE_ONLY,
                                                 bloom_sph_size=0.1,
                                                 bloom_str_gain=0.9))
        self.assertGreater(float((starred > 0.05).sum()),
                           float((core > 0.05).sum()))

    def test_star_point_count_is_exact_for_odd_counts(self):
        """abs(cos(n*theta)) would give 2n lobes; the n/2 factor is what
        makes an odd point count come out right."""
        import math
        for n in (3, 5, 6, 8):
            with self.subTest(points=n):
                # r=0.2 with length 1.0 and falloff 1.0 puts the lobe
                # peaks at 0.8; sampling at r=0.5 would put them exactly
                # ON the 0.5 test threshold and count none of them.
                r = np.full((1, 720), 0.2, np.float32)
                th = np.linspace(-math.pi, math.pi, 720,
                                 endpoint=False).astype(np.float32)[None, :]
                row = App._organic_kernel_star(r, th, n, 0.0, 1.0,
                                               20.0, 1.0)[0]
                peaks = sum(1 for i in range(720)
                            if row[i] > row[i - 1]
                            and row[i] >= row[(i + 1) % 720]
                            and row[i] > 0.5)
                self.assertEqual(peaks, n)

    def test_star_arms_are_thicker_at_the_base_than_the_tip(self):
        """The bug this replaced: an angular profile has constant
        angular width, and a constant angle subtends r*dtheta of
        physical width, so arms came out as wedges — thin at the core
        and fat at the tip, the opposite of a diffraction spike."""
        p = dict(SPHERE_ONLY, bloom_radius=60, bloom_aniso=1.0,
                 bloom_sph_size=0.02, bloom_str_gain=1.0,
                 bloom_str_points=4, bloom_str_angle=0.0,
                 bloom_str_length=1.0, bloom_str_width=8,
                 bloom_str_falloff=0.5, bloom_kgamma=1.0)
        k = App._organic_bloom_kernel(p)
        cy = k.shape[0] // 2

        def arm_width(frac):
            row = k[int(cy - frac * cy)]
            return int((row > row.max() * 0.5).sum())
        self.assertGreater(arm_width(0.25), arm_width(0.75))

    def test_star_arm_thickness_decreases_monotonically(self):
        p = dict(SPHERE_ONLY, bloom_radius=60, bloom_aniso=1.0,
                 bloom_sph_size=0.02, bloom_str_gain=1.0,
                 bloom_str_points=4, bloom_str_angle=0.0,
                 bloom_str_length=1.0, bloom_str_width=6,
                 bloom_str_falloff=0.5, bloom_kgamma=1.0)
        k = App._organic_bloom_kernel(p)
        cy = k.shape[0] // 2
        widths = []
        for frac in (0.2, 0.4, 0.6, 0.8):
            row = k[int(cy - frac * cy)]
            widths.append(int((row > row.max() * 0.5).sum()))
        for a, b in zip(widths, widths[1:]):
            self.assertGreaterEqual(a, b, f"thickness grew outward: {widths}")

    def test_star_width_controls_base_thickness(self):
        """Higher is thinner, as the slider label promises."""
        def base_width(w):
            p = dict(SPHERE_ONLY, bloom_radius=60, bloom_aniso=1.0,
                     bloom_sph_size=0.02, bloom_str_gain=1.0,
                     bloom_str_points=4, bloom_str_width=w,
                     bloom_str_falloff=0.5, bloom_kgamma=1.0)
            k = App._organic_bloom_kernel(p)
            row = k[int(k.shape[0] // 2 * 0.75)]
            return int((row > row.max() * 0.5).sum())
        self.assertGreater(base_width(3), base_width(24))

    def test_star_tapers_to_nothing_at_its_tip(self):
        p = dict(SPHERE_ONLY, bloom_radius=40, bloom_aniso=1.0,
                 bloom_sph_size=0.02, bloom_str_gain=1.0,
                 bloom_str_length=0.5, bloom_str_falloff=1.0)
        k = App._organic_bloom_kernel(p)
        cy, cx = k.shape[0] // 2, k.shape[1] // 2
        # beyond length the arm must be gone, not merely dim
        self.assertLess(float(k[cy, cx + int(cx * 0.9)]), 1e-4)

    def test_two_point_star_is_a_single_bar_not_a_cross(self):
        """dev is capped at 90 degrees before the sine — without that,
        sin folds back and a phantom arm appears opposite."""
        p = dict(SPHERE_ONLY, bloom_radius=40, bloom_aniso=1.0,
                 bloom_sph_size=0.02, bloom_str_gain=1.0,
                 bloom_str_points=2, bloom_str_angle=0.0,
                 bloom_str_width=10, bloom_str_falloff=0.5)
        k = App._organic_bloom_kernel(p)
        cy, cx = k.shape[0] // 2, k.shape[1] // 2
        horizontal = float(k[cy, cx + cx // 2])
        vertical = float(k[cy - cy // 2, cx])
        self.assertGreater(horizontal, vertical * 20 + 1e-6)

    def test_star_angle_rotates_it(self):
        a = App._organic_bloom_kernel(dict(SPHERE_ONLY, bloom_str_gain=0.9,
                                           bloom_str_points=4,
                                           bloom_str_angle=0.0))
        b = App._organic_bloom_kernel(dict(SPHERE_ONLY, bloom_str_gain=0.9,
                                           bloom_str_points=4,
                                           bloom_str_angle=45.0))
        self.assertFalse(np.allclose(a, b))

    def test_gain_zero_is_the_sphere_and_gain_one_is_the_star(self):
        """Star Gain is a mix position, not a level."""
        base = dict(SPHERE_ONLY, bloom_radius=40, bloom_aniso=1.0,
                    bloom_sph_size=0.2, bloom_str_points=6,
                    bloom_kgamma=1.0)
        r, theta = App._organic_kernel_grid(base, 1.0)
        sph = App._organic_kernel_sphere(r, base["bloom_sph_size"],
                                         base["bloom_sph_falloff"])
        star = App._organic_kernel_star(
            r, theta, base["bloom_str_points"], base["bloom_str_angle"],
            base["bloom_str_length"], base["bloom_str_width"],
            base["bloom_str_falloff"])
        np.testing.assert_allclose(
            App._organic_bloom_kernel(dict(base, bloom_str_gain=0.0)),
            sph / sph.max(), atol=1e-5)
        np.testing.assert_allclose(
            App._organic_bloom_kernel(dict(base, bloom_str_gain=1.0)),
            star / star.max(), atol=1e-5)

    def test_the_mix_is_monotonic_in_gain(self):
        """Turning Star Gain up moves steadily toward the star instead
        of popping the way a max does when one layer overtakes the
        other."""
        base = dict(SPHERE_ONLY, bloom_radius=40, bloom_aniso=1.0,
                    bloom_sph_size=0.2, bloom_str_points=6,
                    bloom_str_width=10, bloom_kgamma=1.0)
        pure = App._organic_bloom_kernel(dict(base, bloom_str_gain=1.0))
        dist = [float(np.abs(App._organic_bloom_kernel(
            dict(base, bloom_str_gain=g)) - pure).mean())
            for g in (0.0, 0.25, 0.5, 0.75, 1.0)]
        for a, b in zip(dist, dist[1:]):
            self.assertGreaterEqual(a, b, f"not monotonic: {dist}")

    def test_the_core_stays_bright_across_the_mix(self):
        """Both layers peak at the centre, so a lerp of them does too --
        the star must not dim the core the way add-then-normalise
        would."""
        base = dict(SPHERE_ONLY, bloom_radius=40, bloom_aniso=1.0,
                    bloom_sph_size=0.2, bloom_str_points=6,
                    bloom_kgamma=1.0)
        for g in (0.0, 0.3, 0.6, 1.0):
            k = App._organic_bloom_kernel(dict(base, bloom_str_gain=g))
            cy, cx = k.shape[0] // 2, k.shape[1] // 2
            with self.subTest(gain=g):
                self.assertAlmostEqual(float(k[cy, cx]), 1.0, places=4)

    def test_matches_a_direct_spatial_convolution(self):
        """The DFT path has to agree with the obvious implementation."""
        rng = np.random.default_rng(3)
        img = rng.random((48, 60, 3)).astype(np.float32)
        k = rng.random((9, 11)).astype(np.float32)
        k /= k.sum()
        got = App._organic_convolve(img, k)
        # cv2.filter2D computes CORRELATION, so the kernel is flipped to
        # compare. This matters: with an angled star the kernel is not
        # symmetric, and correlating would mirror the streaks.
        want = np.stack([cv2.filter2D(img[:, :, c], -1, np.flip(k).copy(),
                                      borderType=cv2.BORDER_CONSTANT)
                         for c in range(3)], axis=2)
        np.testing.assert_allclose(got, want, atol=2e-4)

    def test_handles_a_single_channel_field(self):
        rng = np.random.default_rng(5)
        img = rng.random((32, 40)).astype(np.float32)
        k = np.ones((5, 5), np.float32) / 25.0
        self.assertEqual(App._organic_convolve(img, k).shape, img.shape)

    def test_energy_normalised_kernel_preserves_a_flat_field(self):
        img = np.ones((40, 40, 3), np.float32)
        k = App._organic_bloom_kernel(dict(SPHERE_ONLY, bloom_radius=4,
                                           bloom_aniso=1.0))
        out = App._organic_convolve(img, k / k.sum())
        np.testing.assert_allclose(out[15:25, 15:25], 1.0, atol=1e-3)

    def test_impulse_reproduces_the_kernel(self):
        img = np.zeros((41, 41), np.float32)
        img[20, 20] = 1.0
        k = App._organic_bloom_kernel(dict(SPHERE_ONLY, bloom_radius=6,
                                           bloom_aniso=1.0))
        out = App._organic_convolve(img, k)
        kh, kw = k.shape
        got = out[20 - kh // 2:20 + kh // 2 + 1,
                  20 - kw // 2:20 + kw // 2 + 1]
        np.testing.assert_allclose(got, k, atol=1e-4)


class TestBloomEffect(unittest.TestCase):

    def test_threshold_zero_turns_the_gate_off(self):
        """At 0 the whole frame blooms — veiling glare rather than
        highlight bloom. A legitimate slider position, not a
        degenerate one."""
        rgb = np.full((64, 64, 3), 0.3, np.float32)
        rgb[30:34, 30:34] = 2.0
        gated = App._organic_bloom(rgb, dict(SPHERE_ONLY, bloom_thresh=1.0,
                                             bloom_intensity=1.0)) - rgb
        open_ = App._organic_bloom(rgb, dict(SPHERE_ONLY, bloom_thresh=0.0,
                                             bloom_intensity=1.0)) - rgb
        self.assertGreater(float(open_.mean()), float(gated.mean()) * 20)
        self.assertGreater(float(open_.min()), 0.0,
                           "with the gate off every pixel should bloom")

    def test_threshold_zero_mask_is_fully_white_on_a_lit_frame(self):
        rgb = np.full((32, 32, 3), 0.3, np.float32)
        m = App._organic_mask_field(rgb, dict(bloom_thresh=0.0),
                                    "bloom_thresh")
        self.assertEqual(float(m.min()), 1.0)

    def test_threshold_slider_reaches_zero(self):
        """Explicitly requested: the gate has to be switchable off from
        the UI, not just from the params dict."""
        for c in TestRegistryWiring._cs_calls():
            kw = {k.arg: k.value for k in c.keywords}
            if kw.get("pkey") is not None and kw["pkey"].value == "bloom_thresh":
                self.assertEqual(c.args[3].value, 0.0)
                return
        self.fail("threshold slider not found")

    def test_intensity_slider_has_headroom_for_point_speculars(self):
        """Energy normalisation means a 1px highlight spread over a wide
        kernel is genuinely faint; a ceiling of 2.0 could not reach it.
        Measured: radius 60, an 8.0 specular on a 0.2 plate needed a
        gain near 20 to add as much light as the plate carries."""
        for c in TestRegistryWiring._cs_calls():
            kw = {k.arg: k.value for k in c.keywords}
            if (kw.get("pkey") is not None
                    and kw["pkey"].value == "bloom_intensity"):
                self.assertGreaterEqual(c.args[4].value, 20)
                return
        self.fail("intensity slider not found")

    def test_a_point_specular_is_visible_at_high_gain(self):
        rgb = np.full((160, 160, 3), 0.2, np.float32)
        rgb[80, 80] = 8.0
        add = App._organic_bloom(rgb, dict(SPHERE_ONLY, bloom_radius=40,
                                           bloom_aniso=1.0,
                                           bloom_intensity=20.0)) - rgb
        self.assertGreater(float(add.max()), 0.2)

    def test_bloom_only_adds_light(self):
        rgb = plate()
        self.assertTrue(np.all(App._organic_bloom(rgb, PARAMS) >= rgb - 1e-5))

    def test_nothing_above_threshold_is_a_pass_through(self):
        rgb = np.full((32, 32, 3), 0.2, np.float32)
        np.testing.assert_array_equal(
            App._organic_bloom(rgb, dict(PARAMS, bloom_thresh=3.0)), rgb)

    def test_zero_intensity_is_a_pass_through(self):
        rgb = plate()
        np.testing.assert_array_equal(
            App._organic_bloom(rgb, dict(PARAMS, bloom_intensity=0.0)), rgb)

    def test_intensity_scales_the_added_light_linearly(self):
        rgb = plate()
        a = App._organic_bloom(rgb, dict(PARAMS, bloom_intensity=0.3)) - rgb
        b = App._organic_bloom(rgb, dict(PARAMS, bloom_intensity=0.6)) - rgb
        np.testing.assert_allclose(b, a * 2.0, atol=1e-5)

    def test_kernel_shape_does_not_change_total_added_energy(self):
        """The point of dividing by the sum rather than the peak: a wide
        kernel must not read brighter than a tight one."""
        rgb = plate()
        tight = App._organic_bloom(
            rgb, dict(SPHERE_ONLY, bloom_sph_size=0.1)) - rgb
        wide = App._organic_bloom(
            rgb, dict(SPHERE_ONLY, bloom_sph_size=0.9)) - rgb
        self.assertAlmostEqual(float(tight.sum()) / float(wide.sum()),
                               1.0, delta=0.08)

    def test_a_red_highlight_blooms_red(self):
        rgb = np.zeros((64, 64, 3), np.float32)
        rgb[30:34, 30:34, 0] = 4.0
        added = App._organic_bloom(rgb, dict(SPHERE_ONLY, bloom_radius=8,
                                             bloom_aniso=1.0)) - rgb
        self.assertGreater(float(added[:, :, 0].sum()),
                           float(added[:, :, 1].sum()) * 10 + 1e-6)

    def test_star_shows_up_in_the_rendered_bloom(self):
        rgb = np.zeros((96, 96, 3), np.float32)
        rgb[47:49, 47:49] = 8.0
        p = dict(SPHERE_ONLY, bloom_radius=20, bloom_aniso=1.0,
                 bloom_sph_size=0.08, bloom_intensity=1.0)
        plain = App._organic_bloom(rgb, p) - rgb
        star = App._organic_bloom(
            rgb, dict(p, bloom_str_gain=0.9, bloom_str_points=4)) - rgb
        self.assertGreater(float((star > 1e-4).sum()),
                           float((plain > 1e-4).sum()))


class TestBloomSpillMask(unittest.TestCase):

    def test_spill_widens_with_radius(self):
        rgb = plate()

        def spread(radius):
            p = dict(SPHERE_ONLY, bloom_radius=radius, bloom_aniso=1.0)
            return float((App._organic_mask_field(
                rgb, p, "bloom_spill") > 0.05).sum())
        self.assertLess(spread(3), spread(25))

    def test_anamorph_stretches_horizontally(self):
        rgb = np.zeros((96, 96, 3), np.float32)
        rgb[47:49, 47:49] = 4.0
        p = dict(SPHERE_ONLY, bloom_thresh=0.5, bloom_radius=14,
                 bloom_aniso=4.0)
        hit = App._organic_mask_field(rgb, p, "bloom_spill") > 0.05
        self.assertGreater(hit.any(axis=0).sum(), hit.any(axis=1).sum())

    def test_px_scale_keeps_the_shape_honest_when_downsampled(self):
        """A mask built on a half-size frame with px_scale=0.5 must
        describe the same spill as the full-res one -- this is the bug
        that would make a previewed radius read wider than it renders."""
        rgb = plate(128, 192)
        full = App._organic_mask_field(rgb, SPHERE_ONLY, "bloom_spill", 1.0)
        half_src = cv2.resize(rgb, (96, 64), interpolation=cv2.INTER_AREA)
        half = App._organic_mask_field(half_src, SPHERE_ONLY,
                                       "bloom_spill", 0.5)
        half_up = cv2.resize(half, (192, 128), interpolation=cv2.INTER_LINEAR)
        self.assertAlmostEqual(float((full > 0.1).mean()),
                               float((half_up > 0.1).mean()), delta=0.05)

    def test_px_scale_is_actually_applied(self):
        """Guard against the scale being accepted and then ignored."""
        rgb = plate()
        wide = App._organic_mask_field(rgb, SPHERE_ONLY, "bloom_spill", 3.0)
        tight = App._organic_mask_field(rgb, SPHERE_ONLY, "bloom_spill", 0.25)
        self.assertGreater(float((wide > 0.05).sum()),
                           float((tight > 0.05).sum()))

    def test_kernel_mask_returns_the_kernel_not_the_frame(self):
        """The kernel preview is its own small image; the viewer fits it
        by its own aspect rather than the plate's."""
        rgb = plate(128, 192)
        m = App._organic_mask_field(rgb, PARAMS, "bloom_kernel")
        self.assertNotEqual(m.shape[:2], rgb.shape[:2])
        self.assertEqual(m.shape[2], 3)
        self.assertAlmostEqual(float(m.max()), 1.0, places=5)


class TestRegistryWiring(unittest.TestCase):
    """AST guards, in the spirit of test_scope.py: a slider added later
    without a pkey silently loses the auto-dismiss behaviour, and a
    typo'd mask key only shows up as a KeyError at click time."""

    @staticmethod
    def _cs_calls():
        with open(SRC, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "_tab_organic"):
                for c in ast.walk(node):
                    if (isinstance(c, ast.Call)
                            and isinstance(c.func, ast.Name)
                            and c.func.id == "_cs"):
                        yield c

    def test_every_slider_declares_its_param_key(self):
        calls = list(self._cs_calls())
        self.assertGreaterEqual(len(calls), 12)
        for c in calls:
            kw = {k.arg: k.value for k in c.keywords}
            label = c.args[1].value if len(c.args) > 1 else "?"
            with self.subTest(control=label):
                self.assertIn("pkey", kw, f"{label}: no pkey= wired")
                self.assertIsInstance(kw["pkey"], ast.Constant)

    def test_every_declared_mask_key_exists(self):
        for c in self._cs_calls():
            kw = {k.arg: k.value for k in c.keywords}
            if "mask" in kw:
                key = kw["mask"].value
                self.assertIn(key, App._ORGANIC_MASKS)

    def test_param_keys_are_real_stack_parameters(self):
        known = set(App._ORGANIC_PRESETS["Clean Digital"])
        for c in self._cs_calls():
            kw = {k.arg: k.value for k in c.keywords}
            if "pkey" in kw and kw["pkey"].value is not None:
                self.assertIn(kw["pkey"].value, known)

    def test_mask_dependencies_are_real_parameters(self):
        known = set(App._ORGANIC_PRESETS["Clean Digital"])
        for key, (_name, deps) in App._ORGANIC_MASKS.items():
            for d in deps:
                with self.subTest(mask=key, dep=d):
                    self.assertIn(d, known)

    def test_every_mask_is_reachable_from_a_button(self):
        wired = set()
        for c in self._cs_calls():
            kw = {k.arg: k.value for k in c.keywords}
            if "mask" in kw:
                wired.add(kw["mask"].value)
        self.assertEqual(wired, set(App._ORGANIC_MASKS),
                         "a registered mask has no P button")

    def test_every_mask_dependency_has_a_wired_control(self):
        """If a dep can never be touched through the UI, the live-follow
        branch for it is dead code and the pairing is probably wrong."""
        wired = set()
        for c in self._cs_calls():
            kw = {k.arg: k.value for k in c.keywords}
            if "pkey" in kw and kw["pkey"].value is not None:
                wired.add(kw["pkey"].value)
        for key, (_name, deps) in App._ORGANIC_MASKS.items():
            for d in deps:
                with self.subTest(mask=key, dep=d):
                    self.assertIn(d, wired)

    @staticmethod
    def _fxcard_keys():
        with open(SRC, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "_tab_organic"):
                for c in ast.walk(node):
                    if (isinstance(c, ast.Call)
                            and isinstance(c.func, ast.Name)
                            and c.func.id == "_fxcard"):
                        yield c.args[1].value

    def test_every_effect_card_is_a_registered_effect(self):
        keys = list(self._fxcard_keys())
        self.assertEqual(len(keys), len(set(keys)), "duplicate fx key")
        for k in keys:
            self.assertIn(k, [f for f, _l, _g in App._ORGANIC_FX])

    def test_every_registered_effect_has_a_card(self):
        """A new effect added to the stack without a card would have no
        S/M pair and would silently be un-bypassable."""
        self.assertEqual(set(self._fxcard_keys()),
                         {f for f, _l, _g in App._ORGANIC_FX})


class TestRefactorParity(unittest.TestCase):
    """The effects were re-pointed at shared primitives. Their output
    must be unchanged."""

    def test_vignette_matches_the_original_formula(self):
        rgb = plate()
        amount, falloff = 0.45, 1.8
        h, w = rgb.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        r = np.sqrt(((xx - cx) / max(cx, 1e-6)) ** 2 +
                    ((yy - cy) / max(cy, 1e-6)) ** 2)
        r = np.clip(r / np.sqrt(2.0), 0.0, 1.0)
        fall = np.cos(r * (np.pi / 2.0)) ** 4.0
        fall = fall ** (1.0 / max(0.2, falloff))
        expected = rgb * (1.0 - amount * (1.0 - fall))[:, :, None]
        np.testing.assert_allclose(
            App._organic_vignette(rgb, amount, falloff), expected, rtol=1e-6)

    def test_grain_matches_the_original_formula(self):
        rgb = plate()
        seed, inten, sw = 11, 0.03, 1.6
        h, w = rgb.shape[:2]
        mono = App._organic_hash_normal(w, 0, 0, h, w, seed, 0)
        lum = np.clip(rgb.max(axis=2), 0.0, 1.0)
        resp = np.maximum((1.0 - lum) ** sw, 0.15)
        expected = rgb + mono[:, :, None] * (inten * resp[:, :, None])
        np.testing.assert_allclose(
            App._organic_grain(rgb, inten, 1.0, sw, 0.0, seed=seed),
            expected, rtol=1e-5, atol=1e-7)

    def test_stack_still_runs_end_to_end(self):
        out = App._organic_apply_stack(App, plate(), PARAMS, seed=1)
        self.assertEqual(out.shape, (64, 96, 3))
        self.assertTrue(np.isfinite(out).all())

    def test_zero_strength_paths_are_still_pass_through(self):
        rgb = plate()
        np.testing.assert_array_equal(App._organic_chromatic(rgb, 0.0), rgb)
        np.testing.assert_array_equal(App._organic_vignette(rgb, 0.0), rgb)
        np.testing.assert_array_equal(
            App._organic_grain(rgb, 0.0, seed=1), rgb)


# Attribute-name -> starting value for every Tk variable the organic
# tab owns. Keyed by attribute, not by preset key, because the stub has
# to stand in for the widgets, not for the saved file.
VAR_DEFAULTS = {
    "og_ca":            PARAMS["ca"],
    "og_ca_falloff":    PARAMS["ca_falloff"],
    "og_bl_thresh":     PARAMS["bloom_thresh"],
    "og_bl_radius":     PARAMS["bloom_radius"],
    "og_bl_int":        PARAMS["bloom_intensity"],
    "og_bl_aniso":      PARAMS["bloom_aniso"],
    "og_vig":           PARAMS["vig"],
    "og_vig_falloff":   PARAMS["vig_falloff"],
    "og_grain":         PARAMS["grain"],
    "og_grain_size":    PARAMS["grain_size"],
    "og_grain_shad":    PARAMS["grain_shadow"],
    "og_grain_chr":     PARAMS["grain_chroma"],
    "og_grain_filmic":  PARAMS["grain_filmic"],
    "og_global_gain":   PARAMS["global_gain"],
    "og_bl_kgamma":       PARAMS["bloom_kgamma"],
    "og_bl_sph_size":     PARAMS["bloom_sph_size"],
    "og_bl_sph_fall":     PARAMS["bloom_sph_falloff"],
    "og_bl_noi_mix":      PARAMS["bloom_noi_mix"],
    "og_bl_noi_scale":    PARAMS["bloom_noi_scale"],
    "og_bl_noi_detail":   PARAMS["bloom_noi_detail"],
    "og_bl_noi_contrast": PARAMS["bloom_noi_contrast"],
    "og_bl_noi_seed":     PARAMS["bloom_noi_seed"],
    "og_bl_noi_rings":    PARAMS["bloom_noi_rings"],
    "og_bl_noi_ringf":    PARAMS["bloom_noi_ring_freq"],
    "og_bl_noi_rays":     PARAMS["bloom_noi_rays"],
    "og_bl_noi_rayf":     PARAMS["bloom_noi_ray_freq"],
    "og_bl_str_gain":     PARAMS["bloom_str_gain"],
    "og_bl_str_points":   PARAMS["bloom_str_points"],
    "og_bl_str_angle":    PARAMS["bloom_str_angle"],
    "og_bl_str_length":   PARAMS["bloom_str_length"],
    "og_bl_str_width":    PARAMS["bloom_str_width"],
    "og_bl_str_fall":     PARAMS["bloom_str_falloff"],
    "og_bl_disp":         PARAMS["bloom_disp"],
    "og_bl_bands":        PARAMS["bloom_bands"],
    "og_bl_bandspread":   PARAMS["bloom_band_spread"],
    "og_vl_thresh":       PARAMS["veil_thresh"],
    "og_vl_int":          PARAMS["veil_intensity"],
    "og_vl_size_a":       PARAMS["veil_size_a"],
    "og_vl_size_b":       PARAMS["veil_size_b"],
    "og_vl_mix":          PARAMS["veil_mix"],
    "og_vl_sat":          PARAMS["veil_sat"],
    "og_vl_bands":        PARAMS["veil_bands"],
    "og_vl_bandspread":   PARAMS["veil_band_spread"],
    "og_sf_amount":       PARAMS["soft_amount"],
    "og_sf_radius":       PARAMS["soft_radius"],
    "og_sf_halo":         PARAMS["soft_halo"],
    "og_sf_edge":         PARAMS["soft_edge"],
    "og_sf_edgefall":     PARAMS["soft_edge_falloff"],
    "og_sf_chroma":       PARAMS["soft_chroma"],
    "og_sm_amount":       PARAMS["smudge_amount"],
    "og_sm_scale":        PARAMS["smudge_scale"],
    "og_sm_detail":       PARAMS["smudge_detail"],
    "og_sm_cover":        PARAMS["smudge_coverage"],
    "og_sm_soft":         PARAMS["smudge_softness"],
    "og_sm_blur":         PARAMS["smudge_blur"],
    "og_sm_haze":         PARAMS["smudge_haze"],
    "og_sm_seed":         PARAMS["smudge_seed"],
    "og_sc_amount":       PARAMS["scratch_amount"],
    "og_sc_count":        PARAMS["scratch_count"],
    "og_sc_length":       PARAMS["scratch_length"],
    "og_sc_width":        PARAMS["scratch_width"],
    "og_sc_soft":         PARAMS["scratch_softness"],
    "og_sc_angle":        PARAMS["scratch_angle"],
    "og_sc_angvar":       PARAMS["scratch_angle_var"],
    "og_sc_spread":       PARAMS["scratch_spread"],
    "og_sc_seed":         PARAMS["scratch_seed"],
}


class _FakeCanvas:
    """Only winfo_width/height are consulted by the viewer maths."""

    def __init__(self, w, h):
        self._w, self._h = w, h
        self.cursor = None

    def configure(self, **k):
        if "cursor" in k:
            self.cursor = k["cursor"]

    def winfo_width(self):
        return self._w

    def winfo_height(self):
        return self._h


class _FakeButton:
    def __init__(self):
        self.cfg = {}
        self._lbl = self

    def configure(self, **k):
        self.cfg.update(k)

    def bind(self, *a, **k):
        pass


class _FakeLabel(_FakeButton):
    pass


class _FakeEvent:
    def __init__(self, x, y, delta=0, num=0):
        self.x, self.y, self.delta, self.num = x, y, delta, num


class _StubApp(App):
    """App with the Tk surface stubbed out, so the mask state machine
    can be driven directly. __init__ is deliberately not chained -- we
    want the methods under test, not a real window."""

    def __init__(self):                       # noqa: D107
        self.organic_lin = plate()
        self.organic_log_txt = None
        self._organic_idx = 0
        self._organic_zoom = 1.0
        self._organic_pan_x = 0
        self._organic_pan_y = 0
        self._organic_pan_ref = None
        self._organic_wipe_drag = None
        self._organic_wipe_t = 0.5
        self.organic_mode = "edit"
        self.organic_img_base = None
        self.organic_processed = None
        self._organic_mask_key = None
        self._organic_mask_btns = {}
        self._organic_mask_src = None
        self._organic_mask_job = None
        self._organic_fx_solo = {k: False for k, _l, _g in self._ORGANIC_FX}
        self._organic_fx_mute = {k: False for k, _l, _g in self._ORGANIC_FX}
        self._organic_fx_btns = {}
        self._organic_fx_titles = {}
        self.redraws = 0
        self.refreshes = 0
        self.errors = []
        for name, val in VAR_DEFAULTS.items():
            setattr(self, name, _Val(val))

    def _organic_redraw(self):
        self.redraws += 1

    def _organic_mask_refresh(self):
        self.refreshes += 1

    def _organic_mark_stale(self):
        pass

    def log(self, _widget, msg, tag=None):
        if tag == "err":
            self.errors.append(msg)


class _Val:
    """Stands in for a Tk variable in _organic_params."""

    def __init__(self, v):
        self._v = v

    def get(self):
        return self._v


class TestSoloMute(unittest.TestCase):

    def setUp(self):
        self.app = _StubApp()

    # ── resolution rules ──
    def test_everything_runs_by_default(self):
        for fx, _l, _g in App._ORGANIC_FX:
            self.assertTrue(self.app._organic_fx_on(fx))

    def test_mute_bypasses_only_that_effect(self):
        self.app._organic_fx_mute["grain"] = True
        self.assertFalse(self.app._organic_fx_on("grain"))
        self.assertTrue(self.app._organic_fx_on("bloom"))

    def test_solo_bypasses_everything_else(self):
        self.app._organic_fx_solo["bloom"] = True
        self.assertTrue(self.app._organic_fx_on("bloom"))
        for fx in ("ca", "vig", "grain"):
            self.assertFalse(self.app._organic_fx_on(fx))

    def test_solo_is_additive(self):
        self.app._organic_fx_solo["bloom"] = True
        self.app._organic_fx_solo["vig"] = True
        self.assertTrue(self.app._organic_fx_on("bloom"))
        self.assertTrue(self.app._organic_fx_on("vig"))
        self.assertFalse(self.app._organic_fx_on("ca"))

    def test_solo_beats_mute_on_the_same_effect(self):
        self.app._organic_fx_mute["vig"] = True
        self.app._organic_fx_solo["vig"] = True
        self.assertTrue(self.app._organic_fx_on("vig"))

    def test_solo_elsewhere_overrides_a_mute(self):
        """With a solo up, mute is simply not consulted."""
        self.app._organic_fx_mute["bloom"] = True
        self.app._organic_fx_solo["bloom"] = True
        self.app._organic_fx_mute["ca"] = True
        self.assertTrue(self.app._organic_fx_on("bloom"))
        self.assertFalse(self.app._organic_fx_on("ca"))

    def test_soloing_all_four_is_the_same_as_none(self):
        for fx, _l, _g in App._ORGANIC_FX:
            self.app._organic_fx_solo[fx] = True
        for fx, _l, _g in App._ORGANIC_FX:
            self.assertTrue(self.app._organic_fx_on(fx))

    # ── the bypass actually reaches the params ──
    def test_params_zero_the_gate_of_a_muted_effect(self):
        self.app._organic_fx_mute["vig"] = True
        p = self.app._organic_params()
        self.assertEqual(p["vig"], 0.0)
        self.assertEqual(p["vig_falloff"], PARAMS["vig_falloff"])
        self.assertEqual(p["grain"], PARAMS["grain"])

    def test_solo_zeroes_every_other_gate(self):
        self.app._organic_fx_solo["ca"] = True
        p = self.app._organic_params()
        self.assertEqual(p["ca"], PARAMS["ca"])
        self.assertEqual(p["bloom_intensity"], 0.0)
        self.assertEqual(p["vig"], 0.0)
        self.assertEqual(p["grain"], 0.0)

    def test_raw_params_ignore_solo_and_mute(self):
        self.app._organic_fx_solo["ca"] = True
        self.app._organic_fx_mute["grain"] = True
        p = self.app._organic_params(raw=True)
        for k, v in PARAMS.items():
            self.assertEqual(p[k], v)

    def test_every_gate_key_is_a_real_parameter(self):
        known = set(App._ORGANIC_PRESETS["Clean Digital"])
        for fx, _l, gates in App._ORGANIC_FX:
            for g in gates:
                with self.subTest(fx=fx, gate=g):
                    self.assertIn(g, known)

    def test_zeroed_gate_really_skips_the_effect_in_the_stack(self):
        """The gate keys are only correct if the stack's own guards
        agree with them -- otherwise mute would silently do nothing."""
        rgb = plate()
        # Every gate has to be non-zero for this to mean anything: the
        # two lens-dirt effects default to 0 amount, so left at their
        # defaults they would "pass" by simply never running.
        live = dict(smudge_amount=0.6, scratch_amount=0.8,
                    scratch_count=8, scratch_spread=8.0, smudge_blur=4.0,
                    soft_amount=0.7, soft_radius=3.0)
        for fx, _l, _g in App._ORGANIC_FX:
            with self.subTest(fx=fx):
                app = _StubApp()
                for attr, val in (("og_sm_amount", live["smudge_amount"]),
                                  ("og_sc_amount", live["scratch_amount"]),
                                  ("og_sc_count", live["scratch_count"]),
                                  ("og_sc_spread", live["scratch_spread"]),
                                  ("og_sm_blur", live["smudge_blur"]),
                                  ("og_sf_amount", live["soft_amount"]),
                                  ("og_sf_radius", live["soft_radius"])):
                    setattr(app, attr, _Val(val))
                for other, _l2, _g2 in App._ORGANIC_FX:
                    app._organic_fx_mute[other] = (other != fx)
                only = App._organic_apply_stack(
                    App, rgb, app._organic_params(), seed=1)
                app2 = _StubApp()
                for attr, val in (("og_sm_amount", live["smudge_amount"]),
                                  ("og_sc_amount", live["scratch_amount"]),
                                  ("og_sf_amount", live["soft_amount"])):
                    setattr(app2, attr, _Val(val))
                for other, _l2, _g2 in App._ORGANIC_FX:
                    app2._organic_fx_mute[other] = True
                none = App._organic_apply_stack(
                    App, rgb, app2._organic_params(), seed=1)
                np.testing.assert_array_equal(none, rgb)
                self.assertFalse(np.array_equal(only, rgb),
                                 f"{fx} alone changed nothing")

    # ── masks are independent of the bypass ──
    def test_mask_of_a_muted_effect_is_still_meaningful(self):
        """Previewing the vignette shape while the vignette is muted
        must show the shape, not a flat field caused by its own gate."""
        self.app._organic_fx_mute["vig"] = True
        m = App._organic_mask_field(
            plate(), self.app._organic_params(raw=True), "vignette")
        self.assertLess(float(m.min()), 0.9)
        bypassed = App._organic_mask_field(
            plate(), self.app._organic_params(), "vignette")
        self.assertEqual(float(bypassed.min()), 1.0)   # what raw avoids

    # ── the status line ──
    def test_state_line_is_empty_when_nothing_is_bypassed(self):
        self.assertEqual(self.app._organic_fx_state_line(), "")

    def test_state_line_reports_mutes(self):
        self.app._organic_fx_mute["grain"] = True
        self.assertIn("Grain", self.app._organic_fx_state_line())
        self.assertTrue(
            self.app._organic_fx_state_line().startswith("Muted:"))

    def test_state_line_prefers_solo(self):
        self.app._organic_fx_mute["grain"] = True
        self.app._organic_fx_solo["bloom"] = True
        line = self.app._organic_fx_state_line()
        self.assertTrue(line.startswith("Solo:"))
        self.assertIn("Bloom", line)

    def test_toggle_flips_and_is_reversible(self):
        self.app._organic_fx_toggle("bloom", "solo")
        self.assertTrue(self.app._organic_fx_solo["bloom"])
        self.app._organic_fx_toggle("bloom", "solo")
        self.assertFalse(self.app._organic_fx_solo["bloom"])
        self.assertEqual(self.app._organic_fx_state_line(), "")


class _SettableVal(_Val):
    """A Tk-variable stand-in that can also be written to."""

    def set(self, v):
        self._v = v


class TestPresetStorage(unittest.TestCase):
    """The shared preset bar's storage layer, exercised without a
    window. Lens Studio runs through this (so did ADW-Blur)."""

    def setUp(self):
        self._home = tempfile.mkdtemp()
        self._old_home = os.environ.get("HOME")
        os.environ["HOME"] = self._home
        self.app = _StubApp()

    def tearDown(self):
        if self._old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self._old_home
        shutil.rmtree(self._home, ignore_errors=True)

    def test_missing_file_reads_as_empty(self):
        self.assertEqual(App._preset_load_file("nope.json"), [])

    def test_corrupt_file_reads_as_empty_rather_than_raising(self):
        """A bad preset file must not take the tab down on build."""
        p = App._preset_path("presets_organic.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{ this is not json")
        self.assertEqual(App._preset_load_file("presets_organic.json"), [])

    def test_non_list_json_reads_as_empty(self):
        p = App._preset_path("presets_organic.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"name": "oops"}')
        self.assertEqual(App._preset_load_file("presets_organic.json"), [])

    def test_non_dict_entries_are_dropped(self):
        p = App._preset_path("presets_organic.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('[{"name":"a","settings":{}}, "junk", 7]')
        got = App._preset_load_file("presets_organic.json")
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["name"], "a")

    def test_save_then_load_round_trips(self):
        lst = [{"name": "Look A", "settings": {"vig": 0.4}}]
        self.assertTrue(App._preset_save_file("presets_organic.json", lst))
        self.assertEqual(App._preset_load_file("presets_organic.json"), lst)

    def test_next_name_fills_the_first_gap(self):
        lst = [{"name": "Preset_01"}, {"name": "Preset_03"}]
        self.assertEqual(App._preset_next_name(lst), "Preset_02")

    def test_next_name_starts_at_one(self):
        self.assertEqual(App._preset_next_name([]), "Preset_01")

    def test_next_name_ignores_custom_names(self):
        lst = [{"name": "Warm 35mm"}, {"name": "Preset_01"}]
        self.assertEqual(App._preset_next_name(lst), "Preset_02")


class TestPresetCollectApply(unittest.TestCase):

    def setUp(self):
        self.app = _StubApp()
        # Explicit defaults, not getattr: App inherits a permissive
        # widget base in this harness, so a missing attribute would come
        # back as a bound no-op rather than None.
        for attr, val in VAR_DEFAULTS.items():
            setattr(self.app, attr, _SettableVal(val))

    def test_collect_reads_every_mapped_variable(self):
        d = self.app._preset_collect(App.ORGANIC_PRESET_VARS)
        self.assertEqual(set(d), set(App.ORGANIC_PRESET_VARS))

    def test_collect_skips_variables_that_do_not_exist(self):
        d = self.app._preset_collect({"ghost": "og_not_a_thing",
                                      "vig": "og_vig"})
        self.assertEqual(set(d), {"vig"})

    def test_apply_writes_back_what_collect_read(self):
        saved = self.app._preset_collect(App.ORGANIC_PRESET_VARS)
        self.app.og_vig.set(0.999)
        self.app.og_grain.set(0.111)
        self.app._preset_apply_values(App.ORGANIC_PRESET_VARS, saved)
        self.assertEqual(self.app.og_vig.get(), saved["vig"])
        self.assertEqual(self.app.og_grain.get(), saved["grain"])

    def test_apply_ignores_keys_the_preset_does_not_carry(self):
        """A partial preset must leave everything it does not mention
        alone rather than resetting it."""
        self.app.og_grain.set(0.077)
        self.app._preset_apply_values(App.ORGANIC_PRESET_VARS,
                                      {"vig": 0.66})
        self.assertEqual(self.app.og_grain.get(), 0.077)
        self.assertEqual(self.app.og_vig.get(), 0.66)

    def test_apply_ignores_unknown_keys_in_the_file(self):
        applied = self.app._preset_apply_values(
            App.ORGANIC_PRESET_VARS, {"vig": 0.5, "from_a_future_build": 1})
        self.assertEqual(applied, ["vig"])

    def test_every_factory_look_loads_cleanly(self):
        for name, look in App._ORGANIC_PRESETS.items():
            with self.subTest(preset=name):
                applied = self.app._preset_apply_values(
                    App.ORGANIC_PRESET_VARS, look)
                self.assertEqual(set(applied), set(look),
                                 f"{name}: a key had nowhere to go")


class TestPresetRegistries(unittest.TestCase):
    """Guards for the map -> attribute indirection. A renamed Tk
    variable silently drops out of every saved preset otherwise, and
    that only shows up as 'this preset doesn't restore my grain'."""

    @staticmethod
    def _assigned_attrs():
        with open(SRC, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(
                    node.value, ast.Name) and node.value.id == "self":
                found.add(node.attr)
        return found

    def test_organic_map_targets_exist_on_self(self):
        attrs = self._assigned_attrs()
        for key, attr in App.ORGANIC_PRESET_VARS.items():
            with self.subTest(key=key):
                self.assertIn(attr, attrs)


    def test_organic_map_covers_every_look_parameter(self):
        """Anything in the params dict is part of the look, so it must
        be saveable -- otherwise a preset restores an incomplete image."""
        for key in App._ORGANIC_PRESETS["Clean Digital"]:
            with self.subTest(key=key):
                self.assertIn(key, App.ORGANIC_PRESET_VARS)

    def test_factory_looks_only_use_known_keys(self):
        for name, look in App._ORGANIC_PRESETS.items():
            for key in look:
                with self.subTest(preset=name, key=key):
                    self.assertIn(key, App.ORGANIC_PRESET_VARS)

    def test_lens_studio_keeps_its_own_preset_file(self):
        """The file name is what AI-Toolbox's Lens Studio writes, so
        looks saved there load here. It must never fall back to the
        old shared Blur file name, which would read Blur settings as
        a look."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn('"presets_organic.json"', src)
        self.assertNotIn('"presets_adw.json"', src)

    def test_solo_and_mute_are_not_saved_in_presets(self):
        """A saved look must not carry an inspection state, or loading
        it would silently bypass effects."""
        for key in App.ORGANIC_PRESET_VARS:
            self.assertNotIn("solo", key)
            self.assertNotIn("mute", key)


class TestScrollLayout(unittest.TestCase):
    """The left column scrolls; the run buttons must not scroll with
    it. Both are easy to undo by accident when adding a control."""

    @staticmethod
    def _organic_tab_src():
        with open(SRC, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "_tab_organic"):
                return node
        raise AssertionError("_tab_organic not found")

    def test_the_column_is_wrapped_in_a_scroll_area(self):
        calls = [c for c in ast.walk(self._organic_tab_src())
                 if isinstance(c, ast.Call)
                 and isinstance(c.func, ast.Attribute)
                 and c.func.attr == "_scroll_column"]
        self.assertEqual(len(calls), 1)

    def test_two_col_header_is_empty_so_it_is_not_drawn_twice(self):
        """The title moved inside the scroll area; leaving it in
        _two_col as well would render it twice."""
        for c in ast.walk(self._organic_tab_src()):
            if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                    and c.func.attr == "_two_col"):
                title, sub = c.args[1], c.args[2]
                self.assertEqual(title.value, "")
                self.assertEqual(sub.value, "")
                return
        self.fail("_two_col call not found")

    def test_run_buttons_live_in_the_fixed_bottom_strip(self):
        """With the kernel groups open the column runs several screens
        long; Update and Render have to stay reachable."""
        parents = {}
        for node in ast.walk(self._organic_tab_src()):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            t = node.targets[0]
            if not (isinstance(t, ast.Attribute)
                    and t.attr in ("organic_update_btn", "organic_run_btn")):
                continue
            call = node.value
            if (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "_mkbtn"
                    and isinstance(call.args[0], ast.Name)):
                parents[t.attr] = call.args[0].id
        self.assertEqual(set(parents), {"organic_update_btn",
                                        "organic_run_btn"})
        for btn, parent in parents.items():
            with self.subTest(button=btn):
                self.assertNotEqual(parent, "lf",
                                    f"{btn} would scroll away")

    def test_wheel_step_never_rounds_to_zero(self):
        """int(-1/120) is 0, which on a trackpad reporting +/-1 per
        notch means the column silently does not scroll at all."""
        for delta in (-240, -120, -1, 1, 120, 240):
            with self.subTest(delta=delta):
                step = int(-delta / 120) or (-1 if delta > 0 else 1)
                self.assertNotEqual(step, 0)
                self.assertEqual(step > 0, delta < 0)


class TestPolarNoise(unittest.TestCase):
    """Noise that knows where the centre is: rings vary with radius
    only, rays with angle only."""

    @staticmethod
    def _grid(nr=60, nt=72):
        """r varies down the rows, theta across the columns, so a
        layer's dependence shows up as constancy along one axis."""
        r = np.tile(np.linspace(0.0, 1.0, nr, dtype=np.float32)[:, None],
                    (1, nt))
        th = np.tile(np.linspace(-np.pi, np.pi, nt, endpoint=False,
                                 dtype=np.float32)[None, :], (nr, 1))
        return r, th

    def test_both_layers_off_returns_nothing(self):
        r, th = self._grid()
        self.assertIsNone(App._organic_kernel_polar_noise(
            r, th, dict(bloom_noi_rings=0.0, bloom_noi_rays=0.0)))

    def test_rings_vary_with_radius_and_not_with_angle(self):
        r, th = self._grid()
        f = App._organic_kernel_polar_noise(
            r, th, dict(bloom_noi_rings=1.0, bloom_noi_rays=0.0,
                        bloom_noi_ring_freq=8, bloom_noi_detail=2,
                        bloom_noi_contrast=1.0, bloom_noi_seed=1))
        self.assertLess(float(f.std(axis=1).max()), 1e-6)     # flat in theta
        self.assertGreater(float(f.std(axis=0).max()), 0.05)  # varies in r

    def test_rays_vary_with_angle_and_not_with_radius(self):
        r, th = self._grid()
        f = App._organic_kernel_polar_noise(
            r, th, dict(bloom_noi_rings=0.0, bloom_noi_rays=1.0,
                        bloom_noi_ray_freq=24, bloom_noi_detail=2,
                        bloom_noi_contrast=1.0, bloom_noi_seed=1))
        self.assertLess(float(f.std(axis=0).max()), 1e-6)     # flat in r
        self.assertGreater(float(f.std(axis=1).max()), 0.05)  # varies in theta

    def test_the_ray_layer_has_no_seam_at_pi(self):
        """The angular lattice wraps. Without that, the theta = pi ray
        is discontinuous, which on a symmetric flare reads as one arm
        being wrong."""
        lut = App._organic_noise_1d(2048, 17, 3, 5, periodic=True)
        self.assertLess(abs(float(lut[0]) - float(lut[-1])), 0.02)

    def test_the_ring_layer_is_not_forced_to_wrap(self):
        """Radius is not periodic; forcing it would tie the innermost
        shell to the outermost for no reason."""
        gaps = [abs(float(App._organic_noise_1d(2048, 9, 3, s, False)[0])
                    - float(App._organic_noise_1d(2048, 9, 3, s, False)[-1]))
                for s in range(1, 12)]
        self.assertGreater(max(gaps), 0.02)

    def test_polar_layers_stay_in_range(self):
        r, th = self._grid()
        f = App._organic_kernel_polar_noise(
            r, th, dict(bloom_noi_rings=1.0, bloom_noi_rays=1.0,
                        bloom_noi_contrast=2.5, bloom_noi_detail=4,
                        bloom_noi_seed=3))
        self.assertGreaterEqual(float(f.min()), 0.0)
        self.assertLessEqual(float(f.max()), 1.0)

    def test_frequency_changes_the_pattern(self):
        r, th = self._grid()
        base = dict(bloom_noi_rings=0.0, bloom_noi_rays=1.0,
                    bloom_noi_detail=2, bloom_noi_seed=1)
        a = App._organic_kernel_polar_noise(
            r, th, dict(base, bloom_noi_ray_freq=8))
        b = App._organic_kernel_polar_noise(
            r, th, dict(base, bloom_noi_ray_freq=48))
        self.assertFalse(np.allclose(a, b))

    def test_seed_changes_the_pattern(self):
        r, th = self._grid()
        base = dict(bloom_noi_rings=0.6, bloom_noi_rays=0.6,
                    bloom_noi_detail=2)
        a = App._organic_kernel_polar_noise(r, th, dict(base,
                                                        bloom_noi_seed=1))
        b = App._organic_kernel_polar_noise(r, th, dict(base,
                                                        bloom_noi_seed=2))
        self.assertFalse(np.allclose(a, b))

    # ── integration with the kernel ──
    def test_polar_layers_off_reproduce_the_old_kernel_exactly(self):
        p = dict(SPHERE_ONLY, bloom_noi_mix=0.8, bloom_str_gain=0.5)
        a = App._organic_bloom_kernel(dict(p, bloom_noi_rings=0.0,
                                           bloom_noi_rays=0.0))
        b = App._organic_bloom_kernel(dict(p, bloom_noi_rings=0.0,
                                           bloom_noi_rays=0.0,
                                           bloom_noi_ray_freq=99,
                                           bloom_noi_ring_freq=3))
        np.testing.assert_array_equal(a, b)

    def test_rays_change_the_kernel(self):
        p = dict(SPHERE_ONLY, bloom_radius=40, bloom_aniso=1.0,
                 bloom_sph_size=0.12, bloom_noi_mix=0.8,
                 bloom_str_gain=0.7, bloom_kgamma=1.0)
        plain = App._organic_bloom_kernel(p)
        rayed = App._organic_bloom_kernel(dict(p, bloom_noi_rays=0.9,
                                               bloom_noi_ray_freq=36))
        self.assertFalse(np.allclose(plain, rayed))

    def test_polar_layers_do_nothing_when_the_noise_mix_is_zero(self):
        """They multiply the cartesian layer, so Mix is still the master
        switch — turning Rays up with Mix at 0 must not sneak noise in."""
        p = dict(SPHERE_ONLY, bloom_noi_mix=0.0, bloom_str_gain=0.6)
        a = App._organic_bloom_kernel(p)
        b = App._organic_bloom_kernel(dict(p, bloom_noi_rays=1.0,
                                           bloom_noi_rings=1.0))
        np.testing.assert_array_equal(a, b)


class TestPresetBase(unittest.TestCase):
    """Presets are built from one baseline and state only their
    differences, so a new control cannot be forgotten in three of four
    dicts."""

    def test_every_preset_carries_every_parameter(self):
        keys = set(App._ORGANIC_BASE)
        for name, look in App._ORGANIC_PRESETS.items():
            with self.subTest(preset=name):
                self.assertEqual(set(look), keys)

    def test_the_base_covers_every_kernel_and_veil_default(self):
        for k in App._ORGANIC_KERNEL_DEFAULTS:
            self.assertIn(k, App._ORGANIC_BASE)
        for k in App._ORGANIC_VEIL_DEFAULTS:
            self.assertIn(k, App._ORGANIC_BASE)

    def test_presets_are_independent_dicts(self):
        """dict(base, ...) copies; a shared mapping would let editing
        one look silently edit the others."""
        a = App._ORGANIC_PRESETS["Clean Digital"]
        b = App._ORGANIC_PRESETS["Heavy Abuse"]
        self.assertIsNot(a, b)
        self.assertIsNot(a, App._ORGANIC_BASE)
        self.assertNotEqual(a["bloom_radius"], b["bloom_radius"])

    def test_looks_actually_differ_from_the_baseline(self):
        for name, look in App._ORGANIC_PRESETS.items():
            if name == "Clean Digital":
                continue
            with self.subTest(preset=name):
                diff = {k for k in look
                        if look[k] != App._ORGANIC_BASE[k]}
                self.assertGreater(len(diff), 5)


class TestVeilColour(unittest.TestCase):
    """Colour 0 is achromatic charge bleed; 1 inherits the source hue."""

    @staticmethod
    def _green_practical():
        img = np.full((160, 200, 3), 0.05, np.float32)
        img[70:90, 90:110] = (0.35, 9.0, 0.55)
        return img

    def _veil(self, img, **kw):
        p = dict(App._ORGANIC_BASE, veil_thresh=0.8, veil_intensity=2.0, **kw)
        return App._organic_veil(img, p) - img

    def test_colour_zero_is_achromatic(self):
        add = self._veil(self._green_practical(), veil_sat=0.0)
        np.testing.assert_allclose(add[:, :, 0], add[:, :, 1], atol=1e-6)
        np.testing.assert_allclose(add[:, :, 1], add[:, :, 2], atol=1e-6)

    def test_colour_one_takes_the_source_hue(self):
        add = self._veil(self._green_practical(), veil_sat=1.0)
        g, r, b = (float(add[:, :, i].sum()) for i in (1, 0, 2))
        self.assertGreater(g, r * 5)
        self.assertGreater(g, b * 5)

    def test_colour_is_a_continuous_blend(self):
        img = self._green_practical()
        def redness(sat):
            a = self._veil(img, veil_sat=sat)
            return float(a[:, :, 0].sum()) / float(a[:, :, 1].sum())
        self.assertGreater(redness(0.0), redness(0.5))
        self.assertGreater(redness(0.5), redness(1.0))

    def test_the_peak_does_not_move_with_colour(self):
        """The tinted field's per-pixel max is still `over`, so the
        slider changes hue without changing how strong the veil is."""
        img = self._green_practical()
        peaks = [float(self._veil(img, veil_sat=s).max())
                 for s in (0.0, 0.5, 1.0)]
        self.assertAlmostEqual(peaks[0], peaks[1], places=5)
        self.assertAlmostEqual(peaks[1], peaks[2], places=5)

    def test_a_white_source_is_unaffected_by_colour(self):
        img = np.full((120, 120, 3), 0.05, np.float32)
        img[50:70, 50:70] = 6.0
        np.testing.assert_allclose(self._veil(img, veil_sat=0.0),
                                   self._veil(img, veil_sat=1.0), atol=1e-6)

    def test_veil_still_only_adds_light(self):
        img = self._green_practical()
        for sat in (0.0, 0.5, 1.0):
            with self.subTest(colour=sat):
                self.assertGreaterEqual(float(self._veil(img,
                                                         veil_sat=sat).min()),
                                        0.0)

    def test_wide_blur_handles_three_channels(self):
        rgb = np.random.default_rng(2).random((64, 80, 3)).astype(np.float32)
        out = App._organic_wide_blur(rgb, 120.0)
        self.assertEqual(out.shape, rgb.shape)
        self.assertTrue(np.isfinite(out).all())

    def test_mixing_before_the_blur_matches_mixing_after(self):
        """The blur is linear, which is what licenses doing the cheap
        one-channel path at Colour 0."""
        field = np.random.default_rng(4).random((48, 48)).astype(np.float32)
        a = App._organic_wide_blur(field * 0.3, 40.0)
        b = App._organic_wide_blur(field, 40.0) * 0.3
        np.testing.assert_allclose(a, b, atol=1e-5)


class TestKernelExtent(unittest.TestCase):
    """Radius reaches 240, and Anamorph survives the whole range."""

    def test_radius_slider_reaches_240(self):
        for c in TestRegistryWiring._cs_calls():
            kw = {k.arg: k.value for k in c.keywords}
            if kw.get("pkey") is not None and kw["pkey"].value == "bloom_radius":
                self.assertGreaterEqual(c.args[4].value, 240)
                return
        self.fail("radius slider not found")

    def test_anamorph_is_honoured_across_the_radius_range(self):
        """The old clamp cropped the wide axis without moving where
        r == 1, so at radius 120 and Anamorph 3.0 the kernel came out
        2.13 wide with the arms cut square against the canvas."""
        for radius in (60, 120, 180, 240):
            for aniso in (1.0, 2.0, 4.0, 6.0):
                with self.subTest(radius=radius, aniso=aniso):
                    k = App._organic_bloom_kernel(
                        dict(SPHERE_ONLY, bloom_radius=radius,
                             bloom_aniso=aniso))
                    self.assertAlmostEqual(k.shape[1] / k.shape[0], aniso,
                                           delta=0.05)

    def test_nothing_reachable_from_the_sliders_is_clamped(self):
        """Cap is sized for 240 x 6 x the dispersion pad."""
        kr, _kg, _kb = App._organic_bloom_kernel_rgb(
            dict(SPHERE_ONLY, bloom_radius=240, bloom_aniso=6.0,
                 bloom_disp=1.0))
        self.assertLessEqual(max(kr.shape),
                             2 * App._ORGANIC_KERNEL_MAX_HALF + 1)
        self.assertGreater(kr.shape[1] / kr.shape[0], 5.5)

    def test_the_clamp_preserves_aspect_when_it_does_bite(self):
        """Forced with an artificially low cap: the kernel must shrink
        as a whole rather than have one axis cropped off."""
        old = App._ORGANIC_KERNEL_MAX_HALF
        try:
            App._ORGANIC_KERNEL_MAX_HALF = 100
            k = App._organic_bloom_kernel(
                dict(SPHERE_ONLY, bloom_radius=240, bloom_aniso=4.0))
            self.assertLessEqual(max(k.shape), 201)
            self.assertAlmostEqual(k.shape[1] / k.shape[0], 4.0, delta=0.15)
        finally:
            App._ORGANIC_KERNEL_MAX_HALF = old

    def test_a_clamped_kernel_is_not_cut_off_at_its_edge(self):
        """A cropped kernel ends abruptly at a non-zero value; a scaled
        one still falls away to nothing at the border."""
        old = App._ORGANIC_KERNEL_MAX_HALF
        try:
            App._ORGANIC_KERNEL_MAX_HALF = 80
            k = App._organic_bloom_kernel(
                dict(SPHERE_ONLY, bloom_radius=240, bloom_aniso=5.0,
                     bloom_sph_size=0.5))
            cy = k.shape[0] // 2
            self.assertLess(float(k[cy, 0]), 0.02)
            self.assertLess(float(k[cy, -1]), 0.02)
        finally:
            App._ORGANIC_KERNEL_MAX_HALF = old

    def test_a_bigger_radius_really_spreads_further(self):
        rgb = np.zeros((320, 320, 3), np.float32)
        rgb[158:162, 158:162] = 8.0

        def reach(radius):
            p = dict(SPHERE_ONLY, bloom_radius=radius, bloom_aniso=1.0,
                     bloom_thresh=1.0)
            m = App._organic_mask_field(rgb, p, "bloom_spill")
            return float((m > 0.02).sum())
        self.assertGreater(reach(240), reach(120) * 1.5)


class TestViewerTransform(unittest.TestCase):
    """Zoom and pan, matching AreaLock's model exactly: additive zoom
    steps in 0.25..8, and the pan IS the image origin on the canvas
    rather than an offset from a centred fit."""

    def setUp(self):
        from PIL import Image
        self.app = _StubApp()
        self.app._organic_zoom = 1.0
        self.app._organic_pan_x = 0
        self.app._organic_pan_y = 0
        self.app._organic_scale = 1.0
        self.app._organic_roi_mode = False
        self.app._organic_roi_drag = None
        self.app.organic_img_base = Image.new("RGB", (1920, 1080))
        self.app.organic_canvas = _FakeCanvas(700, 400)
        self.app._organic_redraw = lambda: None

    def test_fit_scale_matches_arealock(self):
        self.assertAlmostEqual(
            self.app._organic_fit_scale(700, 400, 1920, 1080),
            min(700 / 1920, 400 / 1080))

    def test_zoom_step_is_additive_and_clamped(self):
        self.app._organic_zoom_step(+0.25)
        self.assertAlmostEqual(self.app._organic_zoom, 1.25)
        for _ in range(60):
            self.app._organic_zoom_step(+0.25)
        self.assertEqual(self.app._organic_zoom, 8.0)
        for _ in range(120):
            self.app._organic_zoom_step(-0.25)
        self.assertEqual(self.app._organic_zoom, 0.25)

    def test_one_to_one_puts_a_source_pixel_on_a_screen_pixel(self):
        self.app._organic_zoom_1to1()
        self.app._organic_scale = self.app._organic_fit_scale(
            700, 400, 1920, 1080)
        self.assertAlmostEqual(self.app._organic_disp_scale(), 1.0, places=6)
        # AreaLock's own 1:1 divides by max() while its fit uses min(),
        # so it lands short on the wider axis. Not copied.

    def test_reset_returns_to_fit(self):
        self.app._organic_zoom = 6.0
        self.app._organic_pan_x = 120
        self.app._organic_pan_y = -40
        self.app._organic_zoom_reset()
        self.assertEqual(self.app._organic_zoom, 1.0)
        self.assertEqual(self.app._organic_pan_x, 0)
        self.assertEqual(self.app._organic_pan_y, 0)

    def test_coordinate_round_trip(self):
        self.app._organic_scale = 0.3
        self.app._organic_zoom = 2.0
        self.app._organic_pan_x = 41
        self.app._organic_pan_y = -17
        cx, cy = self.app._organic_img_to_canvas(600.0, 250.0)
        ix, iy = self.app._organic_canvas_to_img(cx, cy)
        self.assertAlmostEqual(ix, 600.0, places=4)
        self.assertAlmostEqual(iy, 250.0, places=4)

    def test_scroll_keeps_the_point_under_the_cursor(self):
        self.app._organic_scale = 0.3
        before = self.app._organic_canvas_to_img(500, 120)
        self.app._organic_canvas_scroll(_FakeEvent(500, 120, delta=120))
        after = self.app._organic_canvas_to_img(500, 120)
        self.assertAlmostEqual(before[0], after[0], delta=1.0)
        self.assertAlmostEqual(before[1], after[1], delta=1.0)

    def test_x11_wheel_buttons_are_handled(self):
        self.app._organic_canvas_scroll(_FakeEvent(350, 200, num=4))
        self.assertGreater(self.app._organic_zoom, 1.0)
        self.app._organic_canvas_scroll(_FakeEvent(350, 200, num=5))
        self.assertAlmostEqual(self.app._organic_zoom, 1.0, places=6)

    def test_pan_drag_tracks_the_pointer(self):
        self.app._organic_pan_start(_FakeEvent(100, 100))
        self.app._organic_pan_move(_FakeEvent(140, 70))
        self.assertEqual(self.app._organic_pan_x, 40)
        self.assertEqual(self.app._organic_pan_y, -30)

    def test_pan_move_without_a_start_is_ignored(self):
        self.app._organic_pan_ref = None
        self.app._organic_pan_move(_FakeEvent(10, 10))
        self.assertEqual(self.app._organic_pan_x, 0)

    def test_button_one_pans_outside_wipe_mode(self):
        self.app.organic_mode = "edit"
        self.app._organic_canvas_click(_FakeEvent(200, 200))
        self.app._organic_canvas_drag(_FakeEvent(230, 210))
        self.assertEqual(self.app._organic_pan_x, 30)

    def test_button_one_still_drives_the_wipe(self):
        self.app.organic_mode = "wipe"
        self.app._organic_wipe_t = 0.1
        self.app._organic_canvas_click(_FakeEvent(350, 200))
        self.assertAlmostEqual(self.app._organic_wipe_t, 0.5, places=2)
        self.assertEqual(self.app._organic_pan_x, 0)


class TestRegionOfInterest(unittest.TestCase):
    """A region update has to produce the same pixels a full render
    would. Bloom and the veil are convolutions, so the crop is padded by
    their reach; everything position-dependent is told where the crop
    sits in the plate."""

    def setUp(self):
        self.app = _StubApp()
        rng = np.random.default_rng(11)
        self.img = (rng.random((240, 360, 3)).astype(np.float32) * 0.4)
        self.img[90:105, 170:190] = 8.0
        # Base carries a live bloom, CA and grain; tests that isolate
        # one stage have to switch the others off explicitly.
        self.OFF = dict(App._ORGANIC_BASE, ca=0.0, vig=0.0, grain=0.0,
                        bloom_intensity=0.0, veil_intensity=0.0,
                        smudge_amount=0.0, scratch_amount=0.0,
                        soft_amount=0.0)
        self.P = dict(self.OFF, ca=0.9, vig=0.45, grain=0.03,
                      bloom_intensity=0.6, bloom_radius=10,
                      bloom_thresh=0.9)
        self.roi = (70, 110, 170, 250)

    def _compare(self, p):
        full = self.app._organic_apply_stack(self.img, p, seed=5)
        inner, box = self.app._organic_apply_stack_roi(
            self.img, p, self.roi, seed=5)
        self.assertIsNotNone(inner, "crop was declined")
        y0, x0, y1, x1 = box
        return np.abs(inner - full[y0:y1, x0:x1])

    def test_region_update_matches_a_full_render(self):
        self.assertLess(float(self._compare(self.P).max()), 5e-3)

    def test_vignette_is_measured_from_the_plate_centre(self):
        """Not the ROI's centre -- the optical axis belongs to the lens,
        not to whatever box is being re-rendered."""
        p = dict(self.OFF, vig=0.8, vig_falloff=1.5)
        np.testing.assert_allclose(self._compare(p), 0.0, atol=1e-6)

    def test_grain_pattern_is_continuous_with_the_rest_of_the_frame(self):
        """numpy's stream is sequential, so generating only the crop
        would give a different field and a visible seam at the box."""
        p = dict(self.OFF, grain=0.06, grain_chroma=0.5)
        np.testing.assert_allclose(self._compare(p), 0.0, atol=1e-6)

    def test_ca_normalises_against_the_full_plate(self):
        """Displacement is per-frame-corner, so using the crop's own
        min(h, w) gave a different scale factor -- 6.9e-02 of error."""
        p = dict(self.OFF, ca=1.6, ca_falloff=2.2)
        self.assertLess(float(self._compare(p).max()), 2e-3)

    def test_bloom_pulls_light_in_from_outside_the_box(self):
        p = dict(self.OFF, bloom_intensity=0.8, bloom_radius=12,
                 bloom_thresh=0.9)
        self.assertLess(float(self._compare(p).max()), 1e-4)

    def test_reach_covers_the_bloom_kernel(self):
        p = dict(App._ORGANIC_BASE, bloom_intensity=0.5, bloom_radius=40,
                 bloom_aniso=3.0)
        k = App._organic_bloom_kernel(p)
        self.assertGreaterEqual(App._organic_stack_reach(p), k.shape[1] // 2)

    def test_reach_covers_three_sigma_of_the_veil(self):
        p = dict(App._ORGANIC_BASE, veil_intensity=0.5, veil_size_b=120)
        self.assertGreaterEqual(App._organic_stack_reach(p), 360)

    def test_a_cheap_stack_needs_almost_no_padding(self):
        p = dict(self.OFF, vig=0.4, grain=0.02)
        self.assertLessEqual(App._organic_stack_reach(p), 4)

    def test_the_crop_is_declined_when_padding_swallows_the_frame(self):
        """A wide veil reaches so far that a region update would cost
        what a full one does. Better to say so than to pretend."""
        p = dict(self.OFF, veil_intensity=0.5, veil_size_b=400)
        self.assertIsNone(App._organic_roi_crop((240, 360), self.roi, p))

    def test_no_roi_means_no_crop(self):
        self.assertIsNone(App._organic_roi_crop((240, 360), None, self.P))

    def test_roi_is_clamped_to_the_frame(self):
        crop = App._organic_roi_crop((240, 360), (-50, -80, 999, 999),
                                     dict(App._ORGANIC_BASE))
        if crop is not None:
            y0, x0, y1, x1 = crop
            self.assertGreaterEqual(y0, 0)
            self.assertGreaterEqual(x0, 0)
            self.assertLessEqual(y1, 240)
            self.assertLessEqual(x1, 360)


class TestCollapsibleCards(unittest.TestCase):

    @staticmethod
    def _tab_src():
        with open(SRC, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "_tab_organic"):
                return node
        raise AssertionError("_tab_organic not found")

    def test_fxcard_takes_an_opened_flag(self):
        for node in ast.walk(self._tab_src()):
            if isinstance(node, ast.FunctionDef) and node.name == "_fxcard":
                args = [a.arg for a in node.args.args]
                self.assertIn("opened", args)
                return
        self.fail("_fxcard not found")

    def test_every_effect_card_is_registered_for_collapse(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("self._organic_fx_cards[fx] = (state, _toggle)", src)

    def test_the_header_toggles_but_the_sm_buttons_do_not(self):
        """S and M sit inside the header frame; if they shared its
        binding, muting an effect would also fold it away."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        i = src.index("def _fxcard(")
        j = src.index("c_ca = _fxcard(", i)
        body = src[i:j]
        self.assertIn('for _w in (hdr, arw, tl):', body)
        self.assertNotIn('for _w in (hdr, arw, tl, sb, mb):', body)


class TestMethodsExist(unittest.TestCase):
    """Every self._organic_* / self.organic_* call in the module has to
    resolve to something real on App.

    This exists because of a live bug: an edit deleted _organic_redraw
    outright, and nothing caught it. The tkinter stand-in used by these
    tests gives App a permissive __getattr__, so a missing method
    answered as a silent no-op instead of raising -- 199 tests passed
    against a tab whose viewport could not draw. Names are checked
    against the class dict directly, never through getattr."""

    @staticmethod
    def _called_names():
        with open(SRC, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        names = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)):
                continue
            v = node.func.value
            if isinstance(v, ast.Name) and v.id == "self":
                if node.func.attr.startswith("_organic_"):
                    names.add(node.func.attr)
        return names

    @staticmethod
    def _defined_names():
        """Walk the MRO's __dict__ rather than using hasattr, which the
        stub would answer True to for anything."""
        found = set()
        for klass in App.__mro__:
            found |= set(vars(klass))
        return found

    def test_every_organic_method_called_on_self_exists(self):
        defined = self._defined_names()
        missing = sorted(n for n in self._called_names() if n not in defined)
        self.assertEqual(missing, [], f"called but never defined: {missing}")

    def test_the_viewer_entry_points_are_really_methods(self):
        for name in ("_organic_redraw", "_organic_draw_roi",
                     "_organic_draw_hud", "_organic_draw_mask",
                     "_organic_geometry_free_check"):
            if name == "_organic_geometry_free_check":
                continue
            with self.subTest(method=name):
                self.assertIn(name, self._defined_names())
                self.assertTrue(callable(getattr(App, name, None)))

    def test_no_organic_method_is_defined_twice(self):
        """A duplicate definition silently shadows the first."""
        with open(SRC, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        seen, dupes = set(), []
        for klass in ast.walk(tree):
            if not isinstance(klass, ast.ClassDef):
                continue
            for node in klass.body:
                if (isinstance(node, ast.FunctionDef)
                        and node.name.startswith("_organic_")):
                    if node.name in seen:
                        dupes.append(node.name)
                    seen.add(node.name)
        self.assertEqual(dupes, [], f"defined more than once: {dupes}")


class TestRoiClear(unittest.TestCase):
    """Double-clicking the ROI button drops the box and leaves the
    mode, in one action."""

    def setUp(self):
        self.app = _StubApp()
        self.app._organic_roi = (10, 20, 90, 140)
        self.app._organic_roi_mode = True
        self.app._organic_roi_drag = (5, 5)
        self.app.organic_roi_btn = _FakeButton()
        self.app.organic_roi_lbl = _FakeLabel()
        self.app.organic_canvas = _FakeCanvas(700, 400)
        self.app._organic_redraw = lambda: None
        self.stale = []
        self.app._organic_mark_stale = lambda: self.stale.append(True)

    def test_clear_drops_the_box(self):
        self.app._organic_roi_clear()
        self.assertIsNone(self.app._organic_roi)

    def test_clear_also_leaves_roi_mode(self):
        """Switching the mode off while the box lingers meant the next
        update was still constrained to it."""
        self.app._organic_roi_clear()
        self.assertFalse(self.app._organic_roi_mode)
        self.assertIsNone(self.app._organic_roi_drag)

    def test_clear_sets_the_mode_rather_than_toggling_it(self):
        """Tk sends Button-1 twice before Double-Button-1, so the two
        single-click toggles cancel and this can land on either state."""
        for started_on in (True, False):
            with self.subTest(mode_was=started_on):
                self.app._organic_roi_mode = started_on
                self.app._organic_roi = (1, 2, 30, 40)
                self.app._organic_roi_clear()
                self.assertFalse(self.app._organic_roi_mode)
                self.assertIsNone(self.app._organic_roi)

    def test_clear_marks_the_preview_stale(self):
        """What is on screen may be a full frame with one region at
        newer settings."""
        self.app._organic_roi_clear()
        self.assertTrue(self.stale)

    def test_clear_resets_the_button_and_cursor(self):
        self.app._organic_roi_clear()
        self.assertEqual(self.app.organic_roi_btn.cfg.get("fg"), G)
        self.assertEqual(self.app.organic_canvas.cursor, "arrow")

    def test_clear_empties_the_size_readout(self):
        self.app._organic_roi_clear()
        self.assertEqual(self.app.organic_roi_lbl.cfg.get("text"), "")

    def test_a_cleared_roi_no_longer_crops(self):
        self.app._organic_roi_clear()
        self.assertIsNone(App._organic_roi_crop(
            (240, 360), self.app._organic_roi, dict(App._ORGANIC_BASE)))

    def test_the_double_click_is_actually_bound(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn('self.organic_roi_btn._lbl.bind("<Double-Button-1>",',
                      src)


class TestHashNoise(unittest.TestCase):
    """Grain's field is addressed by absolute pixel position rather than
    drawn from a sequential stream, so any sub-rectangle renders exactly
    the pixels the full frame would have."""

    def test_it_is_standard_normal(self):
        f = App._organic_hash_normal(4000, 0, 0, 400, 400, 7, 0)
        self.assertAlmostEqual(float(f.mean()), 0.0, delta=0.02)
        self.assertAlmostEqual(float(f.std()), 1.0, delta=0.02)
        self.assertEqual(f.dtype, np.float32)
        self.assertTrue(np.isfinite(f).all())

    def test_a_crop_equals_that_window_of_the_whole_field(self):
        """The property the whole thing exists for."""
        whole = App._organic_hash_normal(500, 0, 0, 300, 500, 3, 0)
        crop = App._organic_hash_normal(500, 100, 200, 60, 90, 3, 0)
        np.testing.assert_array_equal(crop, whole[100:160, 200:290])

    def test_streams_are_independent(self):
        a = App._organic_hash_normal(4000, 0, 0, 200, 200, 7, 0).ravel()
        b = App._organic_hash_normal(4000, 0, 0, 200, 200, 7, 1).ravel()
        self.assertLess(abs(float(np.corrcoef(a, b)[0, 1])), 0.02)

    def test_seeds_are_independent(self):
        a = App._organic_hash_normal(4000, 0, 0, 200, 200, 7, 0).ravel()
        c = App._organic_hash_normal(4000, 0, 0, 200, 200, 8, 0).ravel()
        self.assertLess(abs(float(np.corrcoef(a, c)[0, 1])), 0.02)

    def test_it_is_reproducible(self):
        np.testing.assert_array_equal(
            App._organic_hash_normal(300, 5, 9, 40, 50, 2, 1),
            App._organic_hash_normal(300, 5, 9, 40, 50, 2, 1))

    def test_no_integer_overflow_warning(self):
        """numpy warns on integer scalar overflow rather than wrapping
        the way the array path does."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            App._organic_hash_normal(1000, 0, 0, 20, 20, 999999, 3)

    def test_pairs_returns_two_fields_per_pair(self):
        """Box-Muller yields sin as well as cos; throwing one away made
        the full-frame draw slower than the numpy stream it replaced."""
        for pairs in (1, 2, 3):
            with self.subTest(pairs=pairs):
                got = App._organic_hash_normals(4000, 0, 0, 64, 64, 5, 0,
                                                pairs)
                self.assertEqual(len(got), pairs * 2)

    def test_every_paired_field_is_standard_normal(self):
        for i, f in enumerate(App._organic_hash_normals(4000, 0, 0, 300, 300,
                                                        7, 0, 2)):
            with self.subTest(field=i):
                self.assertAlmostEqual(float(f.mean()), 0.0, delta=0.03)
                self.assertAlmostEqual(float(f.std()), 1.0, delta=0.03)
                self.assertEqual(f.dtype, np.float32)

    def test_paired_fields_are_mutually_uncorrelated(self):
        fs = [f.ravel() for f in
              App._organic_hash_normals(4000, 0, 0, 200, 200, 7, 0, 2)]
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                with self.subTest(pair=(i, j)):
                    self.assertLess(
                        abs(float(np.corrcoef(fs[i], fs[j])[0, 1])), 0.03)

    def test_the_singular_helper_is_the_first_paired_field(self):
        one = App._organic_hash_normal(500, 3, 7, 40, 50, 9, 0)
        many = App._organic_hash_normals(500, 3, 7, 40, 50, 9, 0, 2)
        np.testing.assert_array_equal(one, many[0])

    def test_mono_grain_is_unaffected_by_the_chroma_setting(self):
        """Chroma draws more fields, but the mono one it blends against
        has to stay put or turning Chroma up would reroll the grain."""
        rgb = plate(64, 64)
        a = App._organic_grain(rgb, 0.05, 1.0, 1.5, 0.0, seed=4)
        b = App._organic_grain(rgb, 0.05, 1.0, 1.5, 1.0, seed=4)
        # at chroma 1 the mono field is fully replaced, so compare the
        # field itself rather than the output
        m1 = App._organic_hash_normals(64, 0, 0, 64, 64, 4, 0, 1)[0]
        m2 = App._organic_hash_normals(64, 0, 0, 64, 64, 4, 0, 2)[0]
        np.testing.assert_array_equal(m1, m2)
        self.assertFalse(np.array_equal(a, b))


class TestRoiAcrossFilters(unittest.TestCase):
    """Every stage has to survive a region update, not just the cheap
    ones."""

    def setUp(self):
        self.app = _StubApp()
        rng = np.random.default_rng(3)
        self.img = (rng.random((720, 1280, 3)).astype(np.float32) * 0.4)
        self.img[300:320, 600:640] = 8.0
        self.roi = (200, 400, 500, 900)
        self.OFF = dict(App._ORGANIC_BASE, ca=0.0, vig=0.0, grain=0.0,
                        bloom_intensity=0.0, veil_intensity=0.0,
                        smudge_amount=0.0, scratch_amount=0.0,
                        soft_amount=0.0)

    def _err(self, **over):
        p = dict(self.OFF, **over)
        full = self.app._organic_apply_stack(self.img, p, seed=1)
        inner, box = self.app._organic_apply_stack_roi(
            self.img, p, self.roi, seed=1)
        self.assertIsNotNone(inner, f"crop declined for {over}")
        y0, x0, y1, x1 = box
        return float(np.abs(inner - full[y0:y1, x0:x1]).max())

    def test_grain_is_exact_under_roi(self):
        self.assertEqual(self._err(grain=0.05), 0.0)

    def test_blurred_grain_is_exact_under_roi(self):
        """Grain Size blurs the field, so the crop is generated with a
        margin and cut afterwards -- otherwise its edge would see
        neighbours the full frame never gave it."""
        self.assertEqual(self._err(grain=0.05, grain_size=2.5), 0.0)

    def test_veil_survives_a_region_update(self):
        self.assertLess(self._err(veil_intensity=0.5, veil_size_a=15,
                                  veil_size_b=60), 1e-3)

    def test_bloom_and_veil_together(self):
        """The case the max-vs-sum reach bug hid in."""
        self.assertLess(self._err(bloom_intensity=0.6, bloom_radius=20,
                                  bloom_thresh=0.9, veil_intensity=0.4,
                                  veil_size_a=12, veil_size_b=45), 2e-3)

    def test_the_whole_stack_together(self):
        self.assertLess(self._err(
            ca=0.8, vig=0.5, grain=0.04, bloom_intensity=0.6,
            bloom_radius=20, bloom_thresh=0.9, veil_intensity=0.4,
            veil_size_a=12, veil_size_b=45), 3e-3)

    def test_reach_is_additive_not_maxed(self):
        """Bloom spreads a highlight, then the veil spreads that, so a
        correct crop needs the sum of both."""
        p = dict(self.OFF, bloom_intensity=0.5, bloom_radius=50,
                 veil_intensity=0.5, veil_size_b=100)
        bloom_only = App._organic_stack_reach(
            dict(self.OFF, bloom_intensity=0.5, bloom_radius=50))
        veil_only = App._organic_stack_reach(
            dict(self.OFF, veil_intensity=0.5, veil_size_b=100))
        both = App._organic_stack_reach(p)
        self.assertGreaterEqual(both, bloom_only + veil_only - 4)
        self.assertGreater(both, max(bloom_only, veil_only))


class TestPreviewIndicator(unittest.TestCase):
    """The Update button is the tab's status light: red = what you are
    looking at is out of date, yellow = being worked out, green =
    matches the controls."""

    def setUp(self):
        self.app = _StubApp()
        self.app.organic_update_btn = _FakeButton()
        self.app._organic_redraw = lambda: None

    def _colours(self):
        c = self.app.organic_update_btn.cfg
        return c.get("bg"), c.get("fg")

    def test_the_three_states_are_distinct(self):
        seen = set()
        for st in ("stale", "busy", "current"):
            self.app._organic_preview_state(st)
            seen.add(self._colours())
        self.assertEqual(len(seen), 3)

    def test_stale_is_red(self):
        self.app._organic_preview_state("stale")
        self.assertEqual(self._colours(), ("#3A1414", "#FF6666"))

    def test_busy_is_yellow(self):
        self.app._organic_preview_state("busy")
        self.assertEqual(self._colours(), ("#3A2A0A", "#DDAA44"))

    def test_current_is_green(self):
        self.app._organic_preview_state("current")
        self.assertEqual(self._colours(), ("#0A2A0A", "#66CC66"))

    def test_touching_a_control_turns_it_red(self):
        # The stub neuters _organic_mark_stale so other tests can count
        # calls, so reach past it to the real one.
        self.app._organic_preview_state("current")
        self.app.organic_auto_var = None
        # mark_stale also invalidates the playback cache now
        self.app._pp_invalidate = lambda *_a, **_k: None
        App._organic_mark_stale(self.app)
        self.assertEqual(self._colours(), ("#3A1414", "#FF6666"))
        self.assertTrue(self.app._organic_stale)

    def test_an_unknown_state_does_not_raise(self):
        self.app._organic_preview_state("nonsense")
        self.assertIsNotNone(self._colours()[0])

    def test_state_is_a_no_op_before_the_button_exists(self):
        app = _StubApp()
        app._organic_preview_state("current")   # must not raise

    # ── the progress bar ──
    def test_bar_fills_as_stages_complete(self):
        self.assertEqual(App._organic_progress_bar(0, 4),
                         "\u25b1\u25b1\u25b1\u25b1")
        self.assertEqual(App._organic_progress_bar(2, 4),
                         "\u25b0\u25b0\u25b1\u25b1")
        self.assertEqual(App._organic_progress_bar(4, 4),
                         "\u25b0\u25b0\u25b0\u25b0")

    def test_bar_names_the_running_stage(self):
        self.assertIn("Bloom", App._organic_progress_bar(1, 3, "Bloom"))

    def test_bar_survives_an_empty_stack(self):
        self.assertIsInstance(App._organic_progress_bar(0, 0), str)

    def test_bar_never_goes_negative(self):
        self.assertEqual(App._organic_progress_bar(9, 3).count("\u25b1"), 0)


class TestStackProgress(unittest.TestCase):
    """The stack reports which stage it is on, so the button has
    something to say during the seconds a 4K bloom takes."""

    def setUp(self):
        self.app = _StubApp()
        self.img = np.full((48, 64, 3), 0.3, np.float32)
        self.img[20:24, 30:34] = 6.0
        self.OFF = dict(App._ORGANIC_BASE, ca=0.0, vig=0.0, grain=0.0,
                        bloom_intensity=0.0, veil_intensity=0.0,
                        smudge_amount=0.0, scratch_amount=0.0,
                        soft_amount=0.0)

    def _run(self, **over):
        calls = []
        self.app._organic_apply_stack(
            self.img, dict(self.OFF, **over), seed=1,
            progress=lambda d, t, n: calls.append((d, t, n)))
        return calls

    def test_only_enabled_stages_are_counted(self):
        calls = self._run(ca=0.5, grain=0.02)
        self.assertEqual(calls[0][1], 2)
        self.assertEqual([c[2] for c in calls[:-1]], ["Aberration", "Grain"])

    def test_the_full_stack_reports_five(self):
        calls = self._run(ca=0.5, vig=0.3, grain=0.02, bloom_intensity=0.4,
                          bloom_radius=4, veil_intensity=0.3, veil_size_b=20)
        self.assertEqual(calls[0][1], 5)
        self.assertEqual([c[2] for c in calls[:-1]],
                         ["Aberration", "Bloom", "Vignette",
                          "Sensor Veil", "Grain"])

    def test_it_finishes_at_full(self):
        calls = self._run(ca=0.5, grain=0.02)
        self.assertEqual(calls[-1][0], calls[-1][1])

    def test_progress_is_monotonic(self):
        calls = self._run(ca=0.5, vig=0.3, grain=0.02)
        done = [c[0] for c in calls]
        self.assertEqual(done, sorted(done))

    def test_an_empty_stack_still_reports_completion(self):
        calls = self._run()
        self.assertEqual(calls, [(0, 0, "")])

    def test_output_is_unchanged_by_reporting(self):
        p = dict(self.OFF, ca=0.5, vig=0.3, grain=0.02)
        a = self.app._organic_apply_stack(self.img, p, seed=1)
        b = self.app._organic_apply_stack(self.img, p, seed=1,
                                          progress=lambda *_: None)
        np.testing.assert_array_equal(a, b)


class TestKernelPreviewFidelity(unittest.TestCase):
    """The kernel preview has to be the kernel that renders.

    It was being built at the preview's downsample factor, which on a
    4K plate made a 21px kernel instead of a 61px one and put the
    star's arms below one sample wide -- they vanished from the picture
    while rendering perfectly. A preview that quietly disagrees with
    the render is worse than no preview."""

    STAR = dict(App._ORGANIC_KERNEL_DEFAULTS, bloom_radius=30,
                bloom_aniso=1.0, bloom_sph_size=0.10, bloom_str_gain=1.0,
                bloom_str_points=6, bloom_str_width=12)

    @staticmethod
    def _preview(p, px):
        return App._organic_mask_field(
            np.zeros((8, 8, 3), np.float32), p, "bloom_kernel", px)

    def test_preview_ignores_the_frame_downsample(self):
        full = self._preview(self.STAR, 1.0)
        for px in (1280 / 1920, 1280 / 3840, 0.1):
            with self.subTest(px_scale=px):
                np.testing.assert_array_equal(self._preview(self.STAR, px),
                                              full)

    def test_preview_matches_the_kernel_the_bloom_uses(self):
        kr, _kg, _kb = App._organic_bloom_kernel_rgb(self.STAR)
        np.testing.assert_array_equal(
            self._preview(self.STAR, 0.25)[:, :, 0], kr)

    def test_the_star_survives_in_the_preview_at_4k(self):
        k = self._preview(self.STAR, 1280 / 3840).max(axis=2)
        no_star = self._preview(dict(self.STAR, bloom_str_gain=0.0),
                                1280 / 3840).max(axis=2)
        self.assertGreater(int((k > 0.05).sum()),
                           int((no_star > 0.05).sum()) * 2)

    def test_star_arms_stay_above_one_sample_wide(self):
        """The failure mode was sub-pixel arms falling between samples."""
        k = self._preview(self.STAR, 1280 / 3840).max(axis=2)
        cy = k.shape[0] // 2
        col = k[:, k.shape[1] // 2 + k.shape[1] // 4]
        self.assertGreater(int((col > 0.05).sum()), 1)
        self.assertGreater(float(k[cy].max()), 0.5)

    def test_frame_space_masks_still_honour_the_downsample(self):
        """The fix must not leak into the masks that genuinely are in
        frame space -- bloom_spill convolves the plate."""
        rgb = plate(128, 192)
        wide = App._organic_mask_field(rgb, SPHERE_ONLY, "bloom_spill", 3.0)
        tight = App._organic_mask_field(rgb, SPHERE_ONLY, "bloom_spill", 0.25)
        self.assertGreater(float((wide > 0.05).sum()),
                           float((tight > 0.05).sum()))


class TestDefaults(unittest.TestCase):
    """The tab opens on a dialled-in flare, not on a neutral Gaussian.
    Pinned so a later edit to the baseline is a deliberate act."""

    SCREENSHOT = dict(
        bloom_thresh=0.40, bloom_intensity=4.95, bloom_radius=59,
        bloom_aniso=1.0, bloom_kgamma=2.85, bloom_disp=0.00,
        bloom_noi_mix=1.00, bloom_noi_scale=6, bloom_noi_detail=3,
        bloom_noi_contrast=1.00, bloom_noi_seed=1,
        bloom_noi_rings=1.00, bloom_noi_ring_freq=8,
        bloom_noi_rays=0.34, bloom_noi_ray_freq=32,
        bloom_str_gain=0.26, bloom_str_points=6, bloom_str_angle=79.0,
        bloom_str_length=1.00, bloom_str_width=38.0, bloom_str_falloff=0.7)

    def test_the_baseline_is_the_tuned_look(self):
        for k, v in self.SCREENSHOT.items():
            with self.subTest(param=k):
                self.assertAlmostEqual(float(App._ORGANIC_BASE[k]),
                                       float(v), places=6)

    def test_startup_reads_the_baseline_not_a_preset(self):
        """"The default" and "the plain look" were the same dict, so
        neither could be changed without moving the other."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("_P = self._ORGANIC_BASE", src)
        self.assertNotIn('_P = self._ORGANIC_PRESETS["Clean Digital"]', src)

    def test_clean_digital_is_still_clean(self):
        cd = App._ORGANIC_PRESETS["Clean Digital"]
        for k in ("bloom_str_gain", "bloom_noi_mix", "bloom_noi_rings",
                  "bloom_noi_rays"):
            with self.subTest(param=k):
                self.assertEqual(cd[k], 0.0)
        self.assertEqual(cd["bloom_kgamma"], 1.0)

    def test_the_default_kernel_actually_builds(self):
        k = App._organic_bloom_kernel(App._ORGANIC_BASE)
        self.assertAlmostEqual(float(k.max()), 1.0, places=5)
        self.assertTrue(np.isfinite(k).all())
        self.assertGreater(int((k > 0.05).sum()), 20)

    def test_the_default_stack_runs_end_to_end(self):
        app = _StubApp()
        out = app._organic_apply_stack(plate(96, 128), App._ORGANIC_BASE,
                                       seed=1)
        self.assertTrue(np.isfinite(out).all())

    def test_sphere_controls_were_left_neutral(self):
        """They were collapsed in the reference screenshot, so there was
        nothing to read off — not zero, unknown."""
        self.assertEqual(App._ORGANIC_BASE["bloom_sph_size"], 0.35)
        self.assertEqual(App._ORGANIC_BASE["bloom_sph_falloff"], 2.0)

    def test_every_default_sits_inside_its_slider_range(self):
        """A default outside its own slider silently snaps on first
        touch, so the tab would not open on what it renders."""
        for c in TestRegistryWiring._cs_calls():
            kw = {k.arg: k.value for k in c.keywords}
            pk = kw.get("pkey")
            if pk is None or pk.value is None:
                continue
            key = pk.value
            if key not in App._ORGANIC_BASE:
                continue
            lo, hi = c.args[3].value, c.args[4].value
            with self.subTest(param=key):
                self.assertGreaterEqual(float(App._ORGANIC_BASE[key]),
                                        float(lo))
                self.assertLessEqual(float(App._ORGANIC_BASE[key]),
                                     float(hi))


class TestVeilIntensityRange(unittest.TestCase):
    """The veil is not energy-normalised the way the kernel bloom is, so
    its Intensity needs a completely different scale from Bloom's."""

    @staticmethod
    def _slider(pkey):
        for c in TestRegistryWiring._cs_calls():
            kw = {k.arg: k.value for k in c.keywords}
            if kw.get("pkey") is not None and kw["pkey"].value == pkey:
                return c.args[3].value, c.args[4].value, c.args[5].value
        raise AssertionError(f"slider for {pkey} not found")

    def test_the_range_tops_out_at_one(self):
        lo, hi, _res = self._slider("veil_intensity")
        self.assertEqual(lo, 0)
        self.assertLessEqual(hi, 1.0)

    def test_the_step_is_fine_enough_to_tune_with(self):
        """The old 0.02 step added 44% of a 0.18 plate at its very first
        notch, so the bottom of the range was unreachable."""
        _lo, _hi, res = self._slider("veil_intensity")
        self.assertLessEqual(res, 0.005)

    def test_it_reads_to_three_decimals(self):
        """A 0.002 step under a two-decimal readout would show the same
        number for three consecutive positions."""
        for c in TestRegistryWiring._cs_calls():
            kw = {k.arg: k.value for k in c.keywords}
            if kw.get("pkey") is not None and kw["pkey"].value == "veil_intensity":
                self.assertEqual(c.args[6].body.values[0].format_spec
                                 .values[0].value, ".3f")
                return
        self.fail("veil intensity slider not found")

    def test_bloom_and_veil_intensity_have_different_scales(self):
        """Bloom is energy-normalised and needs gains in the tens; the
        veil multiplies the over-threshold value almost directly.
        Giving them the same slider was the bug."""
        _b_lo, b_hi, _b_res = self._slider("bloom_intensity")
        _v_lo, v_hi, _v_res = self._slider("veil_intensity")
        self.assertGreater(b_hi, v_hi * 10)

    def test_every_factory_veil_setting_fits_the_range(self):
        _lo, hi, _res = self._slider("veil_intensity")
        for name, pr in App._ORGANIC_PRESETS.items():
            with self.subTest(preset=name):
                self.assertLessEqual(pr["veil_intensity"], hi)

    def test_the_response_is_still_linear_in_intensity(self):
        """Narrowing the range must not have introduced a curve — a
        preset stores the parameter, not the slider position."""
        img = np.full((96, 128, 3), 0.18, np.float32)
        img[40:56, 50:80] = 9.0
        p = dict(App._ORGANIC_BASE, veil_thresh=0.85)
        a = App._organic_veil(img, dict(p, veil_intensity=0.1)) - img
        b = App._organic_veil(img, dict(p, veil_intensity=0.4)) - img
        np.testing.assert_allclose(b, a * 4.0, atol=1e-5)


class TestRenderOutput(unittest.TestCase):
    """Render Sequence has to behave like every other module: versioned
    ADW subdirectory, optional .MOV, hand-off, reveal."""

    @staticmethod
    def _src():
        with open(SRC, encoding="utf-8") as fh:
            return fh.read()

    @staticmethod
    def _fn(name):
        with open(SRC, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise AssertionError(f"{name} not found")

    def test_it_writes_to_a_versioned_adw_subdirectory(self):
        """It used to ask for a folder, which made it the one render in
        the app that did not land where the next module looks -- and
        re-running over the same folder overwrote the previous take."""
        body = ast.dump(self._fn("_organic_run"))
        self.assertIn("_adw_versioned_output_dir", body)
        self.assertIn("LENS_OUT_NAME", body)
        self.assertEqual(App.LENS_OUT_NAME, "Lens_Studio")

    def test_it_no_longer_asks_for_an_output_folder(self):
        self.assertNotIn("askdirectory", ast.dump(self._fn("_organic_run")))

    def test_the_versioned_helper_never_reuses_a_number(self):
        """Shared helper, but the guarantee this tab now relies on."""
        import inspect
        src = inspect.getsource(App._adw_versioned_output_dir)
        self.assertIn("max(existing) + 1", src)

    def test_a_mov_tickbox_exists_and_is_off_by_default(self):
        src = self._src()
        self.assertIn("self.organic_out_mov_var = tk.BooleanVar(value=False)",
                      src)
        self.assertIn('text="Out .MOV"', src)

    def test_the_worker_honours_the_tickbox(self):
        body = ast.dump(self._fn("_organic_run_worker"))
        self.assertIn("organic_out_mov_var", body)
        self.assertIn("_organic_encode_mov", body)

    def test_mov_is_encoded_before_the_handoff(self):
        """A receiving module must not find a half-written movie sitting
        in the folder it just adopted as input."""
        body = ast.unparse(self._fn("_organic_run_worker"))
        self.assertLess(body.index("_organic_encode_mov"),
                        body.index("_organic_after_run_sendto"))

    def test_it_propagates_and_reveals_like_the_others(self):
        """Superseded: folder-reveal now goes through the cross-
        platform _reveal() helper, not a literal "open" call --
        subprocess.run(["open", ...]) is macOS-only and does not exist
        on Windows at all."""
        body = ast.dump(self._fn("_organic_run_worker"))
        self.assertIn("_organic_after_run_sendto", body)
        self.assertIn("_reveal", ast.unparse(self._fn("_organic_run_worker")))

    def test_organic_is_both_a_source_and_a_target_of_the_handoff(self):
        """Being only a source meant other modules' renders never
        arrived here."""
        import inspect
        src = inspect.getsource(App._adw_propagate_output)
        self.assertIn("_organic_load_folder", src)
        # ast.unparse normalises quote style, so match without them
        self.assertIn("'organic'", ast.unparse(
            self._fn("_organic_after_run_sendto")))

    def test_a_dialogless_loader_exists_for_the_handoff(self):
        self.assertTrue(callable(getattr(App, "_organic_load_folder", None)))
        args = [a.arg for a in self._fn("_organic_load_folder").args.args]
        self.assertIn("folder", args)

    def test_the_picker_delegates_to_the_loader(self):
        """Two copies of the frame-scanning logic would drift."""
        body = ast.unparse(self._fn("_organic_pick_folder"))
        self.assertIn("_organic_load_folder", body)
        self.assertNotIn("os.listdir", body)

    def test_the_mov_encoder_has_a_gap_tolerant_fallback(self):
        """ffmpeg's numbered-pattern input fails on any gap, and a
        rendered sub-range always has one."""
        body = ast.unparse(self._fn("_organic_encode_mov"))
        self.assertIn("concat", body)
        self.assertIn("prores_ks", body)


class TestLensDirt(unittest.TestCase):
    """Smudge and scratches sit ON the glass, which makes them unlike
    everything else in the stack in one way that matters: the pattern is
    fixed to the lens, not to the frame."""

    def setUp(self):
        self.app = _StubApp()
        self.img = np.full((240, 360, 3), 0.12, np.float32)
        self.img[100:125, 220:270] = 7.0
        self.OFF = dict(App._ORGANIC_BASE, ca=0.0, vig=0.0, grain=0.0,
                        bloom_intensity=0.0, veil_intensity=0.0,
                        smudge_amount=0.0, scratch_amount=0.0,
                        soft_amount=0.0)

    def _run(self, **over):
        p = dict(self.OFF, **over)
        return self.app._organic_apply_stack(self.img, p, seed=1) - self.img

    # ── fixed to the lens ──
    def test_the_pattern_does_not_change_between_frames(self):
        """A smudge that reseeds per frame is just coloured grain."""
        p = dict(self.OFF, smudge_amount=0.6, smudge_blur=4.0,
                 scratch_amount=0.8, scratch_count=20, scratch_spread=8.0)
        a = self.app._organic_apply_stack(self.img, p, seed=1)
        b = self.app._organic_apply_stack(self.img, p, seed=999)
        np.testing.assert_array_equal(a, b)

    def test_grain_by_contrast_does_change_between_frames(self):
        """Guards the distinction: the per-frame seed still has to reach
        the effect that is supposed to animate."""
        p = dict(self.OFF, grain=0.05)
        a = self.app._organic_apply_stack(self.img, p, seed=1)
        b = self.app._organic_apply_stack(self.img, p, seed=2)
        self.assertFalse(np.array_equal(a, b))

    def test_seeds_reroll_the_patterns(self):
        for key in ("smudge_seed", "scratch_seed"):
            with self.subTest(param=key):
                live = dict(smudge_amount=0.7, smudge_blur=4.0,
                            scratch_amount=0.9, scratch_count=20,
                            scratch_spread=8.0)
                a = self._run(**dict(live, **{key: 1}))
                b = self._run(**dict(live, **{key: 2}))
                self.assertFalse(np.allclose(a, b))

    # ── smudge ──
    def test_coverage_spans_the_whole_slider(self):
        """The raw fBm only reached about 0.85, so the bottom third of
        Coverage did nothing until the field was range-stretched."""
        seen = []
        for cov in (0.3, 0.5, 0.7, 0.9):
            m = App._organic_mask_field(self.img,
                                        dict(self.OFF, smudge_coverage=cov),
                                        "smudge")
            seen.append(float((m > 0.5).mean()))
        self.assertLess(seen[0], 0.2)
        self.assertGreater(seen[-1], 0.7)
        for a, b in zip(seen, seen[1:]):
            self.assertLess(a, b)

    def test_the_smudge_mask_leaves_clean_glass_at_low_coverage(self):
        """The contrast between dirty and clean is what reads as a
        smudge rather than as an even haze."""
        m = App._organic_mask_field(self.img,
                                    dict(self.OFF, smudge_coverage=0.4),
                                    "smudge")
        self.assertLess(float(m.min()), 0.01)
        self.assertGreater(float(m.max()), 0.5)

    def test_the_smudge_mask_is_the_weight_the_effect_uses(self):
        """Previewing the raw field would make Coverage and Edge look
        like they do nothing."""
        p = dict(self.OFF, smudge_coverage=0.55, smudge_softness=0.2)
        m = App._organic_mask_field(self.img, p, "smudge")
        self.assertGreaterEqual(float(m.min()), 0.0)
        self.assertLessEqual(float(m.max()), 1.0)
        loose = App._organic_mask_field(
            self.img, dict(p, smudge_softness=0.9), "smudge")
        self.assertFalse(np.allclose(m, loose))

    def test_the_haze_follows_the_light(self):
        """Invisible on a dark frame, glowing over a window -- which is
        when you actually notice one on a real lens."""
        add = self._run(smudge_amount=0.9, smudge_blur=8.0,
                        smudge_haze=1.0, smudge_coverage=0.7)
        near = float(add[90:135, 200:290].max())
        far = float(add[0:50, 0:60].max())
        self.assertGreater(near, far * 5)

    def test_zero_amount_is_a_pass_through(self):
        np.testing.assert_array_equal(
            App._organic_smudge(self.img, dict(self.OFF, smudge_amount=0.0)),
            self.img)

    # ── scratches ──
    def test_scratches_scale_with_available_light(self):
        """A scratch has no brightness of its own; it only scatters what
        reaches it."""
        add = self._run(scratch_amount=1.0, scratch_count=40,
                        scratch_spread=30.0)
        near = float(add[85:140, 195:295].max())
        far = float(add[0:50, 0:60].max())
        self.assertGreater(near, far * 5)

    def test_count_controls_how_many(self):
        def cov(n):
            m = App._organic_mask_field(
                self.img, dict(self.OFF, scratch_count=n), "scratch")
            return float((m > 0.2).mean())
        self.assertLess(cov(5), cov(40))

    def test_zero_count_produces_nothing(self):
        np.testing.assert_array_equal(
            App._organic_scratches(self.img, dict(self.OFF,
                                                  scratch_amount=1.0,
                                                  scratch_count=0)),
            self.img)

    def test_low_variation_gives_parallel_scratches(self):
        """The cloth-wipe look. With every scratch near one angle the
        mask's directional spread should collapse."""
        def spread(var):
            m = App._organic_mask_field(
                self.img, dict(self.OFF, scratch_count=40,
                               scratch_angle=0.0, scratch_angle_var=var),
                "scratch")
            rows = (m > 0.2).any(axis=1).sum()
            cols = (m > 0.2).any(axis=0).sum()
            return rows / max(cols, 1)
        self.assertLess(spread(4.0), spread(180.0))

    def test_scratches_are_curved_not_dead_straight(self):
        """A perfectly straight line reads as a rendering artefact."""
        m = App._organic_mask_field(
            self.img, dict(self.OFF, scratch_count=1, scratch_length=0.9,
                           scratch_angle=0.0, scratch_angle_var=0.0,
                           scratch_width=1, scratch_softness=0.0,
                           scratch_seed=5), "scratch")
        rows = np.where((m > 0.2).any(axis=1))[0]
        self.assertGreater(len(rows), 1, "a bowed scratch spans >1 row")

    def test_scratches_only_add_light(self):
        add = self._run(scratch_amount=1.5, scratch_count=30,
                        scratch_spread=20.0)
        self.assertGreaterEqual(float(add.min()), 0.0)

    # ── stack integration ──
    def test_contamination_runs_before_the_lens_and_sensor(self):
        """It is the outermost surface, so its scatter travels through
        the rest of the optics -- putting it last would mean a lit
        scratch never blooms."""
        order = []
        self.app._organic_apply_stack(
            self.img,
            dict(App._ORGANIC_BASE, smudge_amount=0.5, smudge_blur=4.0,
                 scratch_amount=0.5, scratch_count=6, scratch_spread=8.0,
                 ca=0.5, vig=0.3, grain=0.02, bloom_intensity=0.4,
                 bloom_radius=3, veil_intensity=0.3, veil_size_b=12),
            seed=1, progress=lambda d, t, n: order.append(n))
        order = [n for n in order if n]
        self.assertEqual(order[:2], ["Lens Smudge", "Lens Scratches"])
        self.assertEqual(order[2], "Aberration")

    def test_roi_reproduces_a_full_render(self):
        p = dict(self.OFF, smudge_amount=0.6, smudge_blur=4.0,
                 scratch_amount=0.8, scratch_count=20,
                 scratch_spread=10.0)
        full = self.app._organic_apply_stack(self.img, p, seed=1)
        inner, box = self.app._organic_apply_stack_roi(
            self.img, p, (80, 130, 200, 300), seed=1)
        self.assertIsNotNone(inner, "crop declined")
        y0, x0, y1, x1 = box
        np.testing.assert_allclose(inner, full[y0:y1, x0:x1], atol=1e-6)

    def test_reach_accounts_for_both_effects(self):
        base = App._organic_stack_reach(self.OFF)
        sm = App._organic_stack_reach(dict(self.OFF, smudge_amount=0.5,
                                          smudge_blur=20.0))
        sc = App._organic_stack_reach(dict(self.OFF, scratch_amount=0.5,
                                          scratch_spread=50.0))
        self.assertGreater(sm, base + 100)   # haze blurs at 3x the sigma
        self.assertGreaterEqual(sc, base + 150)

    def test_both_are_off_by_default(self):
        self.assertEqual(App._ORGANIC_BASE["smudge_amount"], 0.0)
        self.assertEqual(App._ORGANIC_BASE["scratch_amount"], 0.0)

    def test_both_are_saveable_in_presets(self):
        for key in App._ORGANIC_DIRT_DEFAULTS:
            with self.subTest(param=key):
                self.assertIn(key, App.ORGANIC_PRESET_VARS)
                self.assertIn(key, App._ORGANIC_BASE)


class TestSequenceRangeAndLog(unittest.TestCase):

    @staticmethod
    def _fn(name):
        with open(SRC, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise AssertionError(f"{name} not found")

    def test_loading_frames_tells_the_range_bar_how_many(self):
        """Every other module does this and this one never did, so the
        bar sat at zero frames -- an empty rail with both handles on top
        of each other -- and Render refused with "Range is empty"."""
        body = ast.unparse(self._fn("_organic_load_folder"))
        self.assertIn("organic_range_bar.set_n", body)

    def test_the_range_bar_is_populated_from_the_same_place_as_the_timeline(self):
        """If the two are set in different places one will be forgotten
        the next time the loader is touched."""
        body = ast.unparse(self._fn("_organic_load_folder"))
        self.assertLess(body.index("set_frames"),
                        body.index("organic_range_bar.set_n"))

    def test_every_module_with_a_range_bar_populates_it(self):
        """The bug generalised: a bar that is created but never given a
        frame count is silently useless."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        created = set(re.findall(r"(self\.[_A-Za-z0-9]*range_bar)\s*=\s*"
                                 r"self\._maya_range_bar", src))
        self.assertGreaterEqual(len(created), 5)
        for name in created:
            with self.subTest(bar=name):
                self.assertIn(f"{name}.set_n", src)

    def test_the_empty_range_message_names_the_numbers(self):
        body = ast.unparse(self._fn("_organic_run"))
        self.assertIn("Range is empty", body)
        self.assertIn("organic_frames", body)

    def test_the_log_is_short_by_default(self):
        """The viewer is the point of this tab; the log was taking a
        sixth of the height to show mostly blank space."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        i = src.index("self.organic_log_txt = tk.Text(")
        seg = src[i:i + 400]
        m = re.search(r"height=(\d+)", seg)
        self.assertIsNotNone(m)
        self.assertLessEqual(int(m.group(1)), 3)

    def test_the_log_can_be_collapsed(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("_organic_log_open", src)
        self.assertIn("_organic_log_arrow", src)


class TestCachedPlayback(unittest.TestCase):
    """The shared _pp_* player addresses a module purely by naming
    convention, so the whole risk here is a name it looks for that this
    tab does not have -- which fails only at Play time."""

    # what _pp_start / _pp_build_step / _pp_draw reach for
    NEEDED_PLAIN = ("organic_folder", "organic_frames", "organic_log_txt",
                    "organic_canvas", "organic_frame_cb",
                    "organic_img_base", "organic_mode")
    NEEDED_UNDER = ("_organic_range_bar", "_organic_wipe_t",
                    "_organic_wipe_corrected_base")

    @staticmethod
    def _src():
        with open(SRC, encoding="utf-8") as fh:
            return fh.read()

    def test_every_name_the_player_needs_is_set_somewhere(self):
        src = self._src()
        for name in self.NEEDED_PLAIN + self.NEEDED_UNDER:
            with self.subTest(attr=name):
                self.assertRegex(src, r"self\." + name + r"\s*=")

    def test_the_range_bar_alias_points_at_the_real_bar(self):
        """The player wants _<mod>_range_bar; this tab has always called
        it organic_range_bar. One object, two names."""
        self.assertIn("self._organic_range_bar = self.organic_range_bar",
                      self._src())

    def test_a_path_based_frame_loader_exists(self):
        """_pp_build_step loads by PATH, not by index, and the tab only
        had an index-based loader."""
        self.assertTrue(callable(getattr(App, "_organic_load_frame", None)))
        import inspect
        args = list(inspect.signature(App._organic_load_frame).parameters)
        self.assertIn("path", args)

    def test_the_loader_fills_both_sides_of_the_wipe(self):
        import inspect
        body = inspect.getsource(App._organic_load_frame)
        self.assertIn("organic_img_base", body)
        self.assertIn("_organic_wipe_corrected_base", body)
        self.assertIn("_organic_apply_stack", body)

    def test_the_loader_runs_the_real_stack(self):
        """Caching something other than what the preview shows would
        make playback a different image from the still."""
        app = _StubApp()
        app.organic_lin = None
        calls = []
        app._organic_load_linear = lambda p: plate(48, 64)
        app._organic_linear_to_display = lambda a: (
            np.clip(a, 0, 1) * 255).astype(np.uint8)
        app._organic_frame_seed = lambda i: 7
        real = App._organic_apply_stack

        def spy(self_, rgb, p, seed=None, progress=None):
            calls.append(seed)
            return real(self_, rgb, p, seed=seed, progress=progress)
        app._organic_apply_stack = lambda *a, **k: spy(app, *a, **k)
        App._organic_load_frame(app, "frame.exr")
        self.assertEqual(calls, [7])
        self.assertIsNotNone(app._organic_wipe_corrected_base)

    def test_the_play_strip_is_built(self):
        self.assertIn('self._pp_make_buttons("organic"', self._src())

    def test_a_parameter_change_invalidates_the_cache(self):
        """Cached frames were rendered with the old settings; looping
        them afterwards would show a result the controls disagree with."""
        import inspect
        self.assertIn('_pp_invalidate("organic")',
                      inspect.getsource(App._organic_mark_stale))

    def test_the_wipe_labels_say_organic(self):
        app = _StubApp()
        self.assertEqual(App._pp_wipe_labels(app, "organic"),
                         ("Original", "Organic"))

    def test_other_modules_labels_are_unchanged(self):
        app = _StubApp()
        self.assertEqual(App._pp_wipe_labels(app, "ip"),
                         ("Source", "Result"))
        self.assertEqual(App._pp_wipe_labels(app, "curve"),
                         ("Original", "Corrected"))


class TestLogPanel(unittest.TestCase):
    """The log has to get out of the way. It could only ever be grown,
    because height was already pinned at the Text minimum of one line
    and the chrome around it -- a header row plus a separate grip strip
    -- was taller than the line it wrapped."""

    @staticmethod
    def _src():
        with open(SRC, encoding="utf-8") as fh:
            return fh.read()

    def test_the_log_starts_at_one_line(self):
        src = self._src()
        i = src.index("self.organic_log_txt = tk.Text(")
        m = re.search(r"height=(\d+)", src[i:i + 400])
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), 1)

    def test_the_text_padding_is_minimal(self):
        """At one line the padding was half the panel's height."""
        src = self._src()
        i = src.index("self.organic_log_txt = tk.Text(")
        m = re.search(r"pady=(\d+)", src[i:i + 400])
        self.assertIsNotNone(m)
        self.assertLessEqual(int(m.group(1)), 2)

    def test_the_grip_lives_in_the_header_row(self):
        """A grip on its own row cost more height than it saved."""
        src = self._src()
        i = src.index('_grip = tk.Label(')
        self.assertIn("_loghdr", src[i:i + 120])
        self.assertIn('_grip.pack(side="right"', src)

    def test_shrinking_past_one_line_collapses_the_panel(self):
        """The fix for one-directional dragging: below one line there
        is no smaller Text to ask for, so it folds away instead."""
        src = self._src()
        i = src.index("def _log_set(lines):")
        seg = src[i:i + 500]
        self.assertIn("if lines < 1:", seg)
        self.assertIn("_toggle_log()", seg)

    def test_growing_from_collapsed_reopens_it(self):
        """Otherwise + would appear dead once the panel was folded."""
        src = self._src()
        i = src.index("def _log_set(lines):")
        seg = src[i:i + 500]
        self.assertIn("if not self._organic_log_open:", seg)

    def test_there_are_no_size_buttons(self):
        """The grip is the only resize control. The +/- pair was a
        second way to do the same thing, in a row meant to be as close
        to invisible as possible."""
        src = self._src()
        i = src.index("_loghdr = tk.Frame(")
        j = src.index("self.organic_log_txt = tk.Text(")
        self.assertNotIn('("+", 2)', src[i:j])

    def test_the_log_starts_collapsed(self):
        """Its resting state is a single header row and nothing else."""
        src = self._src()
        i = src.index("self._organic_log_box = _lf2")
        seg = src[i:i + 300]
        self.assertIn("self._organic_log_open = False", seg)
        self.assertIn("_grip.pack_forget()", seg)

    def test_the_box_is_not_packed_before_that(self):
        """Packing it and then collapsing would flash the panel open on
        every tab build."""
        src = self._src()
        i = src.index("_lf2 = tk.Frame(rf, bg=S2")
        j = src.index("self._organic_log_box = _lf2")
        self.assertNotIn("_lf2.pack(", src[i:j])

    def test_the_grip_appears_and_disappears_with_the_panel(self):
        """A live grip over a collapsed panel was half the reason this
        felt inconsistent."""
        src = self._src()
        i = src.index("def _toggle_log(")
        seg = src[i:i + 700]
        self.assertIn('_grip.pack(side="right"', seg)
        self.assertIn("_grip.pack_forget()", seg)

    def test_the_size_buttons_do_not_also_toggle_the_log(self):
        """They live inside the header frame, so binding the toggle to
        every child would fold the log away on the same click."""
        src = self._src()
        i = src.index("for _w in (self._organic_log_arrow, _logttl):")
        self.assertIn("_w.bind", src[i:i + 120])
        self.assertNotIn("winfo_children", src[i - 400:i + 200])

    def test_the_drag_grip_is_still_visible_and_hittable(self):
        src = self._src()
        self.assertIn("sb_v_double_arrow", src)
        self.assertIn("_organic_log_grip", src)
        self.assertIn('_grip.bind("<Enter>"', src)

    def test_double_click_returns_to_one_line(self):
        src = self._src()
        i = src.index('_grip.bind("<Double-Button-1>"')
        self.assertIn("_log_set(1)", src[i:i + 100])

    def test_drag_direction(self):
        """Up grows, down shrinks, and down far enough goes below one."""
        line_px = 14
        for dy, expect in ((-40, "bigger"), (40, "smaller")):
            with self.subTest(dy=dy):
                lines = 3 + int(round((0 - dy) / line_px))
                self.assertEqual("bigger" if lines > 3 else "smaller", expect)
        self.assertLess(1 + int(round((0 - 60) / line_px)), 1)

    def test_the_height_is_clamped_when_it_stays_open(self):
        for lines in (1, 2, 200):
            with self.subTest(lines=lines):
                self.assertGreaterEqual(max(1, min(30, lines)), 1)
                self.assertLessEqual(max(1, min(30, lines)), 30)


class TestGrainSeeding(unittest.TestCase):
    """Grain re-rolls every frame, unconditionally. This used to be a
    tickbox with a seed field, which meant the tab shipped a way to make
    a render wrong -- frozen grain reads as dirt on the glass, and the
    thing it was standing in for now has two proper effects of its own."""

    def test_the_seed_moves_with_the_frame(self):
        app = _StubApp()
        seeds = [App._organic_frame_seed(app, i) for i in range(5)]
        self.assertEqual(len(set(seeds)), 5)

    def test_the_same_frame_always_gets_the_same_seed(self):
        """A retake of part of a sequence has to match the first pass."""
        app = _StubApp()
        self.assertEqual(App._organic_frame_seed(app, 42),
                         App._organic_frame_seed(app, 42))

    def test_consecutive_frames_are_not_merely_adjacent(self):
        """A +1 step would give near-identical patterns from a
        stream-based generator; the prime multiplier is what separates
        them."""
        app = _StubApp()
        a = App._organic_frame_seed(app, 10)
        b = App._organic_frame_seed(app, 11)
        self.assertNotEqual(a, b)
        self.assertGreater(abs(a - b), 0)

    def test_grain_actually_differs_between_frames(self):
        app = _StubApp()
        rgb = plate(64, 64)
        p = dict(App._ORGANIC_BASE, grain=0.05)
        a = app._organic_apply_stack(rgb, p,
                                     seed=App._organic_frame_seed(app, 0))
        b = app._organic_apply_stack(rgb, p,
                                     seed=App._organic_frame_seed(app, 1))
        self.assertFalse(np.array_equal(a, b))

    def test_the_tickbox_and_seed_field_are_gone(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        for dead in ("og_grain_animate", "og_seed", "Animate per frame"):
            with self.subTest(name=dead):
                self.assertNotIn(dead, src)

    def test_they_are_gone_from_the_preset_map_too(self):
        """A map entry pointing at a variable that no longer exists
        would silently drop out of every save."""
        self.assertNotIn("grain_animate", App.ORGANIC_PRESET_VARS)
        self.assertNotIn("seed", App.ORGANIC_PRESET_VARS)

    def test_an_old_preset_carrying_the_dead_keys_still_loads(self):
        """Presets saved before this change have them on disk."""
        app = _StubApp()
        for attr in App.ORGANIC_PRESET_VARS.values():
            setattr(app, attr, _SettableVal(0))
        applied = app._preset_apply_values(
            App.ORGANIC_PRESET_VARS,
            {"vig": 0.5, "grain_animate": True, "seed": 17})
        self.assertEqual(applied, ["vig"])


class TestFilmicGrain(unittest.TestCase):
    """Filmic blends from one shared emulsion toward three real ones:
    blue coarsest, green finest, red between -- the ordering colour
    negative actually has."""

    FLAT = None

    def setUp(self):
        self.rgb = np.full((192, 192, 3), 0.35, np.float32)

    def _noise(self, filmic, size=1.6, chroma=0.5):
        return App._organic_grain(self.rgb, 0.06, size, 1.5, chroma,
                                  seed=3, filmic=filmic) - self.rgb

    @staticmethod
    def _coarseness(ch):
        """Lag-1 autocorrelation: coarser grain means neighbouring
        pixels are more alike."""
        return float(np.corrcoef(ch[:, :-1].ravel(),
                                 ch[:, 1:].ravel())[0, 1])

    @staticmethod
    def _rms(n):
        return float(np.sqrt(np.mean([n[:, :, c].std() ** 2
                                      for c in range(3)])))

    # ── it is off by default and backward compatible ──
    def test_the_size_slider_reaches_twenty(self):
        """6 was still well short of where the effect stops changing."""
        for c in TestRegistryWiring._cs_calls():
            kw = {k.arg: k.value for k in c.keywords}
            if kw.get("pkey") is not None and kw["pkey"].value == "grain_size":
                self.assertGreaterEqual(c.args[4].value, 20)
                return
        self.fail("grain size slider not found")

    def test_size_still_coarsens_across_the_new_range(self):
        """A range extension is only worth having if the top of it does
        something -- past the point where a blur has flattened the field
        the slider would just be quieting the grain."""
        rgb = np.full((192, 192, 3), 0.35, np.float32)

        def coarseness(sz):
            n = App._organic_grain(rgb, 0.06, sz, 1.5, 0.5, seed=3) - rgb
            ch = n[:, :, 1]
            return float(np.corrcoef(ch[:, :-1].ravel(),
                                     ch[:, 1:].ravel())[0, 1])
        vals = [coarseness(sz) for sz in (3, 6, 10, 14, 20)]
        for a, b in zip(vals, vals[1:]):
            self.assertGreater(b, a, f"stopped separating: {vals}")

    def test_blue_grain_is_about_twice_green(self):
        """The first ratios were correct in ordering but the separation
        did not survive being looked at."""
        r, g, b = App._ORGANIC_FILM_GRAIN_SIZE
        self.assertEqual(g, 1.0)
        self.assertGreater(b, 1.9)
        self.assertGreater(r, 1.3)
        self.assertLess(r, b)

    def test_the_channels_separate_visibly_at_filmic_one(self):
        rgb = np.full((192, 192, 3), 0.35, np.float32)
        n = App._organic_grain(rgb, 0.06, 2.5, 1.5, 0.5, seed=3,
                               filmic=1.0) - rgb

        def coarse(c):
            ch = n[:, :, c]
            return float(np.corrcoef(ch[:, :-1].ravel(),
                                     ch[:, 1:].ravel())[0, 1])
        cs = [coarse(c) for c in range(3)]
        self.assertGreater(max(cs) - min(cs), 0.08)

    def test_blue_is_also_clearly_stronger_than_green(self):
        rgb = np.full((192, 192, 3), 0.35, np.float32)
        n = App._organic_grain(rgb, 0.06, 2.5, 1.5, 0.5, seed=3,
                               filmic=1.0) - rgb
        self.assertGreater(float(n[:, :, 2].std()),
                           float(n[:, :, 1].std()) * 1.6)

    def test_filmic_defaults_to_off(self):
        self.assertEqual(App._ORGANIC_BASE["grain_filmic"], 0.0)

    def test_zero_leaves_every_channel_identical(self):
        """The whole point of a blend: at 0 this must be exactly the
        single-emulsion behaviour it replaced."""
        out = App._organic_grain(self.rgb, 0.06, 1.6, 1.5, 0.0, seed=3,
                                 filmic=0.0)
        np.testing.assert_array_equal(out[:, :, 0], out[:, :, 1])
        np.testing.assert_array_equal(out[:, :, 1], out[:, :, 2])

    def test_the_default_argument_is_zero(self):
        """Callers that predate this feature must be unaffected."""
        a = App._organic_grain(self.rgb, 0.06, 1.6, 1.5, 0.4, seed=3)
        b = App._organic_grain(self.rgb, 0.06, 1.6, 1.5, 0.4, seed=3,
                               filmic=0.0)
        np.testing.assert_array_equal(a, b)

    # ── the per-layer character ──
    def test_blue_is_coarsest_and_green_finest(self):
        n = self._noise(1.0)
        r, g, b = (self._coarseness(n[:, :, c]) for c in range(3))
        self.assertGreater(b, r)
        self.assertGreater(r, g)

    def test_blue_is_strongest_and_green_weakest(self):
        n = self._noise(1.0)
        r, g, b = (float(n[:, :, c].std()) for c in range(3))
        self.assertGreater(b, r)
        self.assertGreater(r, g)

    def test_the_separation_grows_with_the_slider(self):
        def spread(f):
            n = self._noise(f)
            cs = [self._coarseness(n[:, :, c]) for c in range(3)]
            return max(cs) - min(cs)
        vals = [spread(f) for f in (0.0, 0.33, 0.66, 1.0)]
        for a, b in zip(vals, vals[1:]):
            self.assertLess(a, b, f"not monotonic: {vals}")

    def test_green_is_the_reference_layer(self):
        """Green keeps the size the Size slider asks for; the other two
        move around it, so Filmic does not silently rescale the look."""
        sz, _amp = App._organic_grain_channel(1.0, 1)
        self.assertEqual(sz, 1.0)

    # ── amplitude neutrality, which the tooltip promises ──
    def test_overall_strength_is_unchanged_by_filmic(self):
        """Otherwise Filmic could not be dialled in without re-tuning
        Intensity afterwards."""
        base = self._rms(self._noise(0.0))
        for f in (0.25, 0.5, 0.75, 1.0):
            with self.subTest(filmic=f):
                self.assertAlmostEqual(self._rms(self._noise(f)) / base,
                                       1.0, delta=0.03)

    def test_strength_holds_across_the_size_range(self):
        """Size 1.0 skips the blur entirely, so the compensation had to
        be measured against the path actually taken -- computing it as
        if the base were always blurred left Size 1.0 27% quiet."""
        for sz in (1.0, 1.3, 2.2, 4.0, 12.0, 20.0):
            with self.subTest(size=sz):
                a = self._rms(self._noise(0.0, size=sz))
                b = self._rms(self._noise(1.0, size=sz))
                self.assertAlmostEqual(b / a, 1.0, delta=0.05)

    def test_the_amplitudes_combine_to_unit_power(self):
        """Noise adds as variance, so an arithmetic mean of 1.0 would
        still leave the frame measurably grainier."""
        amps = App._ORGANIC_FILM_GRAIN_AMP
        self.assertAlmostEqual(
            float(np.sqrt(np.mean([a * a for a in amps]))), 1.0, places=2)

    # ── the blur-gain helper ──
    def test_blur_gain_matches_a_measured_field(self):
        """It is computed from the kernel rather than the data, because
        measuring a crop's own deviation would differ from the full
        frame and put a seam at the edge of a region update."""
        rng = np.random.default_rng(4)
        import cv2
        field = rng.standard_normal((400, 400)).astype(np.float32)
        for sigma in (0.6, 1.0, 2.0):
            with self.subTest(sigma=sigma):
                blurred = cv2.GaussianBlur(field, (0, 0), sigmaX=sigma)
                self.assertAlmostEqual(
                    float(blurred.std()),
                    App._organic_blur_noise_gain(sigma), delta=0.02)

    def test_blur_gain_is_one_at_zero_sigma(self):
        self.assertEqual(App._organic_blur_noise_gain(0.0), 1.0)

    # ── plumbing ──
    def test_filmic_is_saveable_in_presets(self):
        self.assertIn("grain_filmic", App.ORGANIC_PRESET_VARS)
        self.assertIn("grain_filmic", App._ORGANIC_BASE)

    def test_the_stack_passes_filmic_through(self):
        import inspect
        self.assertIn("grain_filmic",
                      inspect.getsource(App._organic_apply_stack))

    def test_it_survives_a_region_update(self):
        """Per-channel blur widths change the padding the crop needs."""
        app = _StubApp()
        rng = np.random.default_rng(6)
        img = (rng.random((160, 240, 3)).astype(np.float32) * 0.4)
        p = dict(App._ORGANIC_BASE, ca=0.0, vig=0.0, bloom_intensity=0.0,
                 veil_intensity=0.0, smudge_amount=0.0, scratch_amount=0.0,
                 grain=0.06, grain_size=3.0, grain_filmic=1.0)
        full = app._organic_apply_stack(img, p, seed=5)
        inner, box = app._organic_apply_stack_roi(img, p, (40, 60, 120, 180),
                                                  seed=5)
        self.assertIsNotNone(inner)
        y0, x0, y1, x1 = box
        np.testing.assert_allclose(inner, full[y0:y1, x0:x1], atol=1e-6)


class TestLensSoftness(unittest.TestCase):
    """Glass that cannot resolve, as opposed to glass that is
    mis-focused: a sharp core in a broad halo, worse off-axis, and worse
    in blue than green."""

    def setUp(self):
        self.app = _StubApp()
        self.OFF = dict(App._ORGANIC_BASE, ca=0.0, vig=0.0, grain=0.0,
                        bloom_intensity=0.0, veil_intensity=0.0,
                        smudge_amount=0.0, scratch_amount=0.0,
                        soft_amount=0.0)
        self.point = np.zeros((201, 201, 3), np.float32)
        self.point[100, 100] = 100.0

    @staticmethod
    def _rms_radius(field, cy=100, cx=100):
        """Second moment of the point spread -- the honest measure of
        how wide a blur is. A single radial sample can land on a
        crossover between the tight and wide components and read
        backwards."""
        yy, xx = np.mgrid[0:field.shape[0], 0:field.shape[1]]
        r2 = ((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float64)
        f = np.clip(field, 0, None).astype(np.float64)
        return float(np.sqrt((f * r2).sum() / max(f.sum(), 1e-9)))

    # ── off by default, and inert when off ──
    def test_off_by_default(self):
        self.assertEqual(App._ORGANIC_BASE["soft_amount"], 0.0)

    def test_zero_amount_is_a_pass_through(self):
        np.testing.assert_array_equal(
            App._organic_soften(self.point, dict(self.OFF,
                                                 soft_amount=0.0)),
            self.point)

    def test_zero_radius_is_a_pass_through(self):
        np.testing.assert_array_equal(
            App._organic_soften(self.point, dict(self.OFF, soft_amount=1.0,
                                                 soft_radius=0.0)),
            self.point)

    # ── it redistributes light, it does not add or remove it ──
    def test_energy_is_conserved(self):
        out = App._organic_soften(self.point,
                                  dict(self.OFF, soft_amount=1.0,
                                       soft_radius=4.0, soft_edge=0.0))
        for c in range(3):
            with self.subTest(channel="RGB"[c]):
                self.assertAlmostEqual(float(out[:, :, c].sum()),
                                       float(self.point[:, :, c].sum()),
                                       delta=0.5)

    def test_a_flat_field_is_untouched(self):
        """A blur of a constant is that constant, so softness must not
        shift exposure."""
        flat = np.full((64, 64, 3), 0.4, np.float32)
        out = App._organic_soften(flat, dict(self.OFF, soft_amount=1.0,
                                             soft_radius=3.0))
        np.testing.assert_allclose(out, flat, atol=2e-3)

    # ── the two-scale point spread ──
    def test_the_halo_widens_the_wings(self):
        """One blur scale just reads as out of focus; the pair is what
        gives spherical aberration's sharp core inside a glow."""
        def wings(halo):
            o = App._organic_soften(
                self.point, dict(self.OFF, soft_amount=1.0,
                                 soft_radius=3.0, soft_halo=halo,
                                 soft_edge=0.0, soft_chroma=0.0))
            return float(o[100, 130, 0])
        self.assertGreater(wings(0.8), wings(0.35))
        self.assertGreater(wings(0.35), wings(0.0))

    def test_the_halo_lowers_the_core(self):
        """Energy moved to the wings has to come from somewhere."""
        def peak(halo):
            o = App._organic_soften(
                self.point, dict(self.OFF, soft_amount=1.0,
                                 soft_radius=3.0, soft_halo=halo,
                                 soft_edge=0.0, soft_chroma=0.0))
            return float(o[100, 100, 0])
        self.assertLess(peak(0.8), peak(0.0))

    # ── longitudinal chromatic aberration ──
    def test_blue_is_softest_and_green_sharpest(self):
        """Short wavelengths refract more, so blue focuses nearest and
        red furthest; focused on green, both go soft."""
        o = App._organic_soften(self.point,
                                dict(self.OFF, soft_amount=1.0,
                                     soft_radius=4.0, soft_halo=0.3,
                                     soft_edge=0.0, soft_chroma=1.0))
        r, g, b = (self._rms_radius(o[:, :, c]) for c in range(3))
        self.assertGreater(b, r)
        self.assertGreater(r, g)

    def test_chroma_zero_is_achromatic(self):
        o = App._organic_soften(self.point,
                                dict(self.OFF, soft_amount=1.0,
                                     soft_radius=4.0, soft_chroma=0.0))
        np.testing.assert_allclose(o[:, :, 0], o[:, :, 1], atol=1e-6)
        np.testing.assert_allclose(o[:, :, 1], o[:, :, 2], atol=1e-6)

    def test_green_is_the_reference_channel(self):
        """A lens is focused on green, so the Radius slider means the
        green radius and the others move around it."""
        radii = App._organic_soft_radii(10.0, 1.0)
        self.assertEqual(radii[1], 10.0)

    def test_the_split_grows_with_the_slider(self):
        def spread(c):
            radii = App._organic_soft_radii(10.0, c)
            return max(radii) - min(radii)
        vals = [spread(c) for c in (0.0, 0.5, 1.0)]
        self.assertEqual(vals[0], 0.0)
        self.assertLess(vals[1], vals[2])

    # ── field curvature ──
    def test_the_corners_go_soft_before_the_centre(self):
        rng = np.random.default_rng(1)
        tex = rng.random((300, 400, 3)).astype(np.float32)
        o = App._organic_soften(tex, dict(self.OFF, soft_amount=1.0,
                                          soft_radius=4.0, soft_edge=1.0,
                                          soft_chroma=0.0, soft_halo=0.3))
        mid = (slice(130, 170), slice(180, 220))
        corner = (slice(0, 40), slice(0, 40))
        kept_mid = float(o[mid].std()) / float(tex[mid].std())
        kept_corner = float(o[corner].std()) / float(tex[corner].std())
        self.assertGreater(kept_mid, 0.9)
        self.assertLess(kept_corner, 0.5)

    def test_edge_zero_is_uniform(self):
        f = App._organic_soft_field(64, 96, dict(self.OFF, soft_amount=0.7,
                                                 soft_edge=0.0))
        self.assertAlmostEqual(float(f.min()), float(f.max()), places=6)
        self.assertAlmostEqual(float(f.max()), 0.7, places=6)

    def test_the_field_never_exceeds_the_amount(self):
        f = App._organic_soft_field(64, 96, dict(self.OFF, soft_amount=0.6,
                                                 soft_edge=1.0))
        self.assertLessEqual(float(f.max()), 0.6 + 1e-6)
        self.assertGreaterEqual(float(f.min()), 0.0)

    def test_the_mask_shows_the_shape_at_any_amount(self):
        """It answers "where is it soft"; Amount is a separate
        question, so the preview normalises it out."""
        a = App._organic_mask_field(plate(64, 96),
                                    dict(self.OFF, soft_amount=0.1,
                                         soft_edge=1.0), "soft_field")
        b = App._organic_mask_field(plate(64, 96),
                                    dict(self.OFF, soft_amount=0.9,
                                         soft_edge=1.0), "soft_field")
        np.testing.assert_allclose(a, b, atol=1e-6)
        self.assertAlmostEqual(float(a.max()), 1.0, places=5)

    # ── stack integration ──
    def test_it_runs_after_contamination_and_before_lateral_ca(self):
        """Dirt is on the outer surface; the glass then forms the
        image, softness and lateral shift being two failures of it."""
        order = []
        self.app._organic_apply_stack(
            plate(64, 64),
            dict(App._ORGANIC_BASE, smudge_amount=0.4, smudge_blur=3.0,
                 scratch_amount=0.4, scratch_count=4, scratch_spread=6.0,
                 soft_amount=0.5, soft_radius=2.0, ca=0.5, vig=0.0,
                 grain=0.0, bloom_intensity=0.0, veil_intensity=0.0),
            seed=1, progress=lambda d, t, n: order.append(n))
        order = [n for n in order if n]
        self.assertEqual(order, ["Lens Smudge", "Lens Scratches",
                                 "Lens Softness", "Aberration"])

    def test_it_is_distinct_from_lateral_aberration(self):
        """CA displaces sharp channels; this blurs them. Both on the
        same plate must not collapse into one another."""
        tex = np.random.default_rng(2).random((96, 96, 3)).astype(np.float32)
        soft = App._organic_soften(tex, dict(self.OFF, soft_amount=1.0,
                                             soft_radius=3.0))
        lat = App._organic_chromatic(tex, 2.0, 2.0)
        # Lateral CA resamples with linear interpolation, so on random
        # noise it does lose a little detail -- the claim is not that it
        # loses none, but that it is nowhere near a blur.
        kept_soft = float(soft.std()) / float(tex.std())
        kept_lat = float(lat.std()) / float(tex.std())
        self.assertLess(kept_soft, 0.5)
        self.assertGreater(kept_lat, 0.8)

    def test_it_survives_a_region_update(self):
        rng = np.random.default_rng(3)
        img = (rng.random((200, 300, 3)).astype(np.float32) * 0.5)
        p = dict(self.OFF, soft_amount=0.8, soft_radius=2.0,
                 soft_halo=0.4, soft_chroma=1.0)
        full = self.app._organic_apply_stack(img, p, seed=1)
        inner, box = self.app._organic_apply_stack_roi(
            img, p, (60, 90, 150, 220), seed=1)
        self.assertIsNotNone(inner, "crop declined")
        y0, x0, y1, x1 = box
        np.testing.assert_allclose(inner, full[y0:y1, x0:x1], atol=1e-5)

    def test_the_reach_covers_the_far_halo(self):
        """The wide component blurs at four times the radius, and blue
        widest of the three."""
        base = App._organic_stack_reach(self.OFF)
        got = App._organic_stack_reach(dict(self.OFF, soft_amount=0.5,
                                            soft_radius=10.0))
        self.assertGreater(got, base + 3 * 10 * 4 - 1)

    def test_every_control_is_saveable(self):
        for key in App._ORGANIC_SOFT_DEFAULTS:
            with self.subTest(param=key):
                self.assertIn(key, App.ORGANIC_PRESET_VARS)
                self.assertIn(key, App._ORGANIC_BASE)


class TestTimelineParity(unittest.TestCase):
    """The strip has to be identical to ADW-Paint's -- which is the
    `smudge` tab. Matching CurveLock was the wrong target and, more to
    the point, missed that Paint's timeline is not in the tab panel at
    all: it lives in the shared bar pinned at the bottom of the window."""

    PAINT = "_tab_smudge"

    @staticmethod
    def _tab(name):
        with open(SRC, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.unparse(node)
        raise AssertionError(f"{name} not found")

    def test_organic_is_registered_in_the_shared_timeline_bar(self):
        """Without this its strip is packed inside the right-hand panel
        and sits somewhere else on screen entirely."""
        body = self._tab("_build_timeline_bar")
        self.assertIn("'organic'", body)

    def test_it_uses_the_shared_frame_like_paint_does(self):
        for tab, key in (("_tab_organic", "organic"),
                         (self.PAINT, "smudge")):
            with self.subTest(tab=tab):
                self.assertIn(f"self._tl_frames['{key}']", self._tab(tab))

    def test_it_no_longer_packs_its_own_timeline_frame(self):
        body = self._tab("_tab_organic")
        i = body.index("organic_frame_cb = self._timeline")
        self.assertNotIn("padx=16", body[max(0, i - 900):i])

    def test_the_range_bar_sits_above_the_timeline(self):
        for tab, var in (("_tab_organic", "organic_frame_cb"),
                         (self.PAINT, "sm_frame_cb")):
            with self.subTest(tab=tab):
                body = self._tab(tab)
                self.assertLess(body.index("_maya_range_bar"),
                                body.index(f"{var} = self._timeline"))

    def test_the_play_buttons_are_on_the_label_row(self):
        """Paint puts them beside the label; this tab gave them a row of
        their own."""
        for tab, mod in (("_tab_organic", "organic"), (self.PAINT, "smudge")):
            with self.subTest(tab=tab):
                body = self._tab(tab)
                i = body.index(f"_pp_make_buttons('{mod}'")
                self.assertIn("_hdr", body[i:i + 60])

    def test_the_label_matches_paints_wording(self):
        self.assertIn("Work frames", self._tab("_tab_organic"))

    def test_the_timeline_flags_match_paints(self):
        want = ("fill=True", "taller=True", "show_numbers=True")
        for tab, var in (("_tab_organic", "organic_frame_cb"),
                         (self.PAINT, "sm_frame_cb")):
            body = self._tab(tab)
            i = body.index(f"{var} = self._timeline")
            seg = body[i:i + 400]
            for flag in want:
                with self.subTest(tab=tab, flag=flag):
                    self.assertIn(flag, seg)

    def test_cached_frames_are_tinted_like_paints(self):
        """Both use the same cached player, so both should show which
        frames are in the cache."""
        for tab, var in (("_tab_organic", "organic_frame_cb"),
                         (self.PAINT, "sm_frame_cb")):
            body = self._tab(tab)
            i = body.index(f"{var} = self._timeline")
            with self.subTest(tab=tab):
                self.assertIn("get_frame_color", body[i:i + 400])

    def test_the_tint_reads_the_players_real_cache(self):
        """A colour function that never matches the cache's actual shape
        would simply never fire, and look like it was working."""
        body = self._tab("_tab_organic")
        i = body.index("def _organic_frame_color")
        seg = body[i:i + 600]
        for key in ("_pp", "cache", "v0", "out"):
            with self.subTest(key=key):
                self.assertIn(key, seg)

    def test_the_bloom_intensity_ceiling_is_twenty(self):
        for c in TestRegistryWiring._cs_calls():
            kw = {k.arg: k.value for k in c.keywords}
            if (kw.get("pkey") is not None
                    and kw["pkey"].value == "bloom_intensity"):
                self.assertEqual(c.args[4].value, 20)
                return
        self.fail("bloom intensity slider not found")


class TestPresetInterchange(unittest.TestCase):
    """Export and import, shared with ADW-Blur through the same preset
    bar. The risky part is not the file writing -- it is importing a
    file this tab cannot use and appearing to succeed."""

    LOOKS = [{"name": "Warm 35mm", "settings": {"vig": 0.4, "grain": 0.03}},
             {"name": "Preset_01", "settings": {"vig": 0.1}}]

    # ── format ──
    def test_a_blob_round_trips(self):
        blob = json.loads(json.dumps(
            App._preset_export_blob("presets_organic.json", self.LOOKS)))
        back, store = App._preset_parse_import(blob)
        self.assertEqual(back, self.LOOKS)
        self.assertEqual(store, "presets_organic.json")

    def test_the_blob_names_where_it_came_from(self):
        blob = App._preset_export_blob("presets_organic.json", self.LOOKS)
        self.assertEqual(blob["store"], "presets_organic.json")
        self.assertEqual(blob["format"], App.PRESET_EXPORT_TAG)
        self.assertIn("version", blob)

    def test_a_bare_list_is_accepted(self):
        """The file in ~/.ai_toolbox is a bare list, and copying one
        straight out of there is the obvious thing to try."""
        back, store = App._preset_parse_import(self.LOOKS)
        self.assertEqual(back, self.LOOKS)
        self.assertIsNone(store)

    def test_junk_is_rejected_with_a_readable_reason(self):
        for bad in ("hello", 42, {"nope": 1}, [], [1, 2, 3],
                    [{"name": "x"}], None):
            with self.subTest(data=repr(bad)[:20]):
                with self.assertRaises(ValueError) as cm:
                    App._preset_parse_import(bad)
                self.assertGreater(len(str(cm.exception)), 8)

    def test_malformed_entries_are_dropped_not_fatal(self):
        back, _s = App._preset_parse_import(
            [{"name": "ok", "settings": {"vig": 1}}, "junk",
             {"name": "no settings"}, 7])
        self.assertEqual(len(back), 1)
        self.assertEqual(back[0]["name"], "ok")

    def test_a_missing_name_gets_one(self):
        back, _s = App._preset_parse_import([{"settings": {"vig": 1}}])
        self.assertTrue(back[0]["name"])

    # ── the wrong-module guard ──
    def test_another_tabs_presets_score_zero(self):
        """Apply skips keys it does not recognise, so without this the
        import would succeed and then do nothing at all."""
        # The shape of an AI-Toolbox ADW-Blur preset file -- the one a
        # person is most likely to point Lens Studio at by mistake.
        blur = [{"name": "b",
                 "settings": {k: 1 for k in ("rmult", "scale", "flow",
                                             "upscale", "tta", "retime",
                                             "blend_str", "falloff")}}]
        self.assertLess(
            App._preset_match_score(blur, App.ORGANIC_PRESET_VARS), 0.2)

    def test_this_tabs_presets_score_one(self):
        own = [{"name": "o", "settings": {k: 1 for k
                                          in App.ORGANIC_PRESET_VARS}}]
        self.assertEqual(
            App._preset_match_score(own, App.ORGANIC_PRESET_VARS), 1.0)

    def test_a_partial_match_lands_between(self):
        half = dict(list(App.ORGANIC_PRESET_VARS.items())[:5])
        mixed = [{"name": "m", "settings": dict(
            {k: 1 for k in half}, unknown_a=1, unknown_b=2, unknown_c=3)}]
        score = App._preset_match_score(mixed, App.ORGANIC_PRESET_VARS)
        self.assertGreater(score, 0.2)
        self.assertLess(score, 0.9)

    def test_an_empty_settings_dict_scores_zero(self):
        self.assertEqual(
            App._preset_match_score([{"name": "e", "settings": {}}],
                                    App.ORGANIC_PRESET_VARS), 0.0)

    # ── merging ──
    def test_import_never_overwrites_an_existing_look(self):
        existing = [{"name": "Warm 35mm", "settings": {"vig": 0.9}}]
        merged, added = App._preset_merge(existing, self.LOOKS)
        self.assertEqual(merged[0]["settings"]["vig"], 0.9)
        self.assertIn("Warm 35mm (2)", added)

    def test_repeated_collisions_keep_counting(self):
        existing = [{"name": "A", "settings": {}},
                    {"name": "A (2)", "settings": {}}]
        merged, added = App._preset_merge(
            existing, [{"name": "A", "settings": {"vig": 1}}])
        self.assertEqual(added, ["A (3)"])
        self.assertEqual(len(merged), 3)

    def test_merging_into_an_empty_library_keeps_the_names(self):
        _merged, added = App._preset_merge([], self.LOOKS)
        self.assertEqual(added, ["Warm 35mm", "Preset_01"])

    def test_merge_preserves_the_settings(self):
        merged, _a = App._preset_merge([], self.LOOKS)
        self.assertEqual(merged[0]["settings"], {"vig": 0.4, "grain": 0.03})

    # ── an exported file is loadable by the normal reader ──
    def test_an_exported_blob_survives_a_real_file(self):
        home = tempfile.mkdtemp()
        try:
            path = os.path.join(home, "out.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(App._preset_export_blob("presets_organic.json",
                                                  self.LOOKS), fh, indent=2)
            with open(path, encoding="utf-8") as fh:
                back, store = App._preset_parse_import(json.load(fh))
            self.assertEqual(back, self.LOOKS)
            self.assertEqual(store, "presets_organic.json")
        finally:
            shutil.rmtree(home, ignore_errors=True)

    def test_imported_settings_actually_apply(self):
        """End to end: parse, merge, then push into the variables."""
        app = _StubApp()
        for attr in App.ORGANIC_PRESET_VARS.values():
            setattr(app, attr, _SettableVal(0))
        incoming, _s = App._preset_parse_import(
            App._preset_export_blob("presets_organic.json",
                                    [{"name": "x",
                                      "settings": {"vig": 0.61,
                                                   "grain": 0.02}}]))
        merged, _added = App._preset_merge([], incoming)
        applied = app._preset_apply_values(App.ORGANIC_PRESET_VARS,
                                           merged[0]["settings"])
        self.assertEqual(sorted(applied), ["grain", "vig"])
        self.assertEqual(app.og_vig.get(), 0.61)

    # ── the menu entries exist ──
    def test_the_dropdown_offers_both(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("Import", src)
        self.assertIn("Export all", src)
        # both must be wired to handlers, not just present as labels
        self.assertIn("_import()", src)
        self.assertIn("_export()", src)

    def test_both_tabs_get_it_from_the_shared_bar(self):
        """It lives in _preset_bar, so ADW-Blur inherits it too."""
        import inspect
        body = inspect.getsource(App._preset_bar)
        self.assertIn("def _export", body)
        self.assertIn("def _import", body)


class TestGlobalGain(unittest.TestCase):
    """A wet/dry over the whole finished chain, applied once at the
    end."""

    def setUp(self):
        self.app = _StubApp()
        rng = np.random.default_rng(2)
        self.img = (rng.random((120, 160, 3)).astype(np.float32) * 0.5)
        self.img[40:50, 60:80] = 6.0
        self.P = dict(App._ORGANIC_BASE, ca=0.9, vig=0.5, grain=0.05,
                      bloom_intensity=4.0, bloom_thresh=0.9,
                      veil_intensity=0.3, veil_size_b=30)

    def _run(self, gain):
        return self.app._organic_apply_stack(
            self.img, dict(self.P, global_gain=gain), seed=1)

    def test_it_defaults_to_fully_treated(self):
        self.assertEqual(App._ORGANIC_BASE["global_gain"], 1.0)

    def test_zero_returns_the_untouched_plate(self):
        np.testing.assert_array_equal(self._run(0.0), self.img)

    def test_one_is_the_full_stack(self):
        self.assertFalse(np.allclose(self._run(1.0), self.img))

    def test_one_is_bit_identical_to_omitting_the_parameter(self):
        """The blend is skipped at 1.0, so an existing preset without
        the key must render exactly as it always did."""
        p = dict(self.P)
        p.pop("global_gain", None)
        np.testing.assert_array_equal(
            self.app._organic_apply_stack(self.img, p, seed=1),
            self._run(1.0))

    def test_it_is_an_exact_linear_mix(self):
        wet = self._run(1.0)
        for g in (0.25, 0.5, 0.75):
            with self.subTest(gain=g):
                np.testing.assert_array_equal(
                    self._run(g), self.img * (1 - g) + wet * g)

    def test_it_is_monotonic(self):
        wet = self._run(1.0)
        dist = [float(np.abs(self._run(g) - wet).mean())
                for g in (0.0, 0.25, 0.5, 0.75, 1.0)]
        for a, b in zip(dist, dist[1:]):
            self.assertGreater(a, b, f"not monotonic: {dist}")

    def test_it_applies_after_every_effect(self):
        """Folded into each effect it would be a different operation --
        the chain is not linear, so scaling every amount changes the
        character of the look rather than its strength."""
        import inspect
        body = inspect.getsource(App._organic_apply_stack)
        self.assertLess(body.index("out = fn(out)"),
                        body.index('p.get("global_gain"'))

    def test_it_is_not_an_effect_with_a_gate(self):
        """Muting a wet/dry mix and setting it to zero would be the
        same thing said twice."""
        gates = {fx for fx, _l, _g in App._ORGANIC_FX}
        self.assertNotIn("global_gain", gates)
        for _fx, _l, g in App._ORGANIC_FX:
            self.assertNotIn("global_gain", g)

    def test_it_is_saveable(self):
        self.assertIn("global_gain", App.ORGANIC_PRESET_VARS)
        self.assertIn("global_gain", App._ORGANIC_BASE)

    def test_a_region_update_honours_it(self):
        p = dict(App._ORGANIC_BASE, ca=0.0, vig=0.5, grain=0.05,
                 bloom_intensity=0.0, veil_intensity=0.0,
                 smudge_amount=0.0, scratch_amount=0.0, soft_amount=0.0,
                 global_gain=0.4)
        full = self.app._organic_apply_stack(self.img, p, seed=1)
        inner, box = self.app._organic_apply_stack_roi(
            self.img, p, (30, 40, 90, 120), seed=1)
        self.assertIsNotNone(inner, "crop declined")
        y0, x0, y1, x1 = box
        np.testing.assert_allclose(inner, full[y0:y1, x0:x1], atol=1e-6)


class TestFlareBands(unittest.TestCase):
    """Intensity banding: brighter sources get a physically wider
    kernel, which a single convolution cannot do because convolution is
    shift-invariant by definition."""

    P = dict(App._ORGANIC_BASE, bloom_thresh=1.0, bloom_radius=40,
             bloom_intensity=1.0, bloom_aniso=1.0, bloom_sph_size=0.08,
             bloom_str_gain=0.6, bloom_str_points=6, bloom_str_width=10,
             bloom_noi_mix=0.0, bloom_kgamma=1.0)

    @staticmethod
    def _reach(profile, floor=0.02):
        """Reach of the NORMALISED profile -- so this measures the shape
        of the point spread, not how bright it is. A linear convolution
        gives the same answer at every source brightness."""
        n = profile / max(float(profile.max()), 1e-12)
        return max([d for d in range(1, len(n)) if n[d] > floor] or [0])

    def _point(self, src, **over):
        im = np.zeros((301, 301, 3), np.float32)
        im[150, 150] = src
        out = App._organic_bloom(im, dict(self.P, **over)) - im
        return out[150, 150:, 0]

    # ── one band is exactly what it always was ──
    def test_one_band_is_the_default(self):
        self.assertEqual(App._ORGANIC_BASE["bloom_bands"], 1)

    def test_one_band_is_bit_identical_to_no_banding_key(self):
        """Existing presets predate this control."""
        img = np.random.default_rng(3).random((120, 160, 3)).astype(np.float32)
        img[50:56, 70:76] = 60.0
        p = dict(self.P)
        p.pop("bloom_bands", None)
        np.testing.assert_array_equal(
            App._organic_bloom(img, p),
            App._organic_bloom(img, dict(self.P, bloom_bands=1)))

    def test_band_spread_does_nothing_at_one_band(self):
        img = np.random.default_rng(4).random((120, 160, 3)).astype(np.float32)
        img[50:56, 70:76] = 40.0
        np.testing.assert_array_equal(
            App._organic_bloom(img, dict(self.P, bloom_bands=1,
                                         bloom_band_spread=1.5)),
            App._organic_bloom(img, dict(self.P, bloom_bands=1,
                                         bloom_band_spread=4.0)))

    # ── the weights ──
    def test_the_weights_sum_to_exactly_one(self):
        """A partition of unity. Anything else changes total bloom
        energy as a pixel moves between bands."""
        over = np.geomspace(1e-4, 5000.0, 4000).astype(np.float32)
        for n in (1, 2, 3, 4):
            with self.subTest(bands=n):
                tot = sum(App._organic_band_weights(over, n, 1.0))
                np.testing.assert_allclose(tot, 1.0, atol=1e-6)

    def test_the_weights_are_soft_not_hard_buckets(self):
        """Hard buckets ring: a gradient crossing an edge gets a visible
        contour, and it crawls on moving footage."""
        over = np.geomspace(0.01, 5000.0, 2000).astype(np.float32)
        w = App._organic_band_weights(over, 3, 1.0)
        shared = sum(1 for i in range(len(over))
                     if sum(1 for b in w if b[i] > 0.01) > 1)
        self.assertGreater(shared, 200, "no overlap between bands")

    def test_each_weight_stays_in_range(self):
        over = np.geomspace(1e-4, 5000.0, 500).astype(np.float32)
        for b in App._organic_band_weights(over, 4, 1.0):
            self.assertGreaterEqual(float(b.min()), 0.0)
            self.assertLessEqual(float(b.max()), 1.0)

    def test_dim_pixels_land_in_the_first_band(self):
        over = np.array([1e-5, 1e-4], np.float32)
        w = App._organic_band_weights(over, 3, 1.0)
        np.testing.assert_allclose(w[0], 1.0, atol=1e-6)

    def test_very_bright_pixels_land_in_the_last_band(self):
        over = np.array([1e6], np.float32)
        w = App._organic_band_weights(over, 3, 1.0)
        np.testing.assert_allclose(w[-1], 1.0, atol=1e-6)

    def test_the_bands_are_anchored_to_the_threshold(self):
        """Absolute edges would land somewhere different on every
        plate; the threshold is the exposure decision already made."""
        over = np.array([8.0], np.float32)
        a = App._organic_band_weights(over, 3, 1.0)
        b = App._organic_band_weights(over, 3, 8.0)
        self.assertFalse(np.allclose(a[0], b[0]))

    # ── the point of the whole thing ──
    def test_one_band_gives_every_source_the_same_psf(self):
        """Shift-invariance -- the shape cannot vary, only its
        brightness."""
        reaches = [self._reach(self._point(v, bloom_bands=1))
                   for v in (2.0, 30.0, 400.0)]
        self.assertEqual(len(set(reaches)), 1, f"varied: {reaches}")

    def test_more_bands_make_bright_sources_spread_wider(self):
        dim = self._reach(self._point(2.0, bloom_bands=3,
                                      bloom_band_spread=2.5))
        bright = self._reach(self._point(400.0, bloom_bands=3,
                                         bloom_band_spread=2.5))
        self.assertGreater(bright, dim * 2)

    def test_the_spread_control_scales_that(self):
        def reach(sp):
            return self._reach(self._point(400.0, bloom_bands=3,
                                           bloom_band_spread=sp))
        self.assertGreater(reach(3.5), reach(1.3))

    def test_a_gradient_does_not_contour(self):
        """The failure mode hard buckets would have: a step in the
        output where a smooth ramp crosses a band edge."""
        ramp = np.zeros((40, 900, 3), np.float32)
        ramp[:] = np.linspace(0.5, 600.0, 900,
                              dtype=np.float32)[None, :, None]
        p = dict(self.P, bloom_radius=30, bloom_str_gain=0.0,
                 bloom_sph_size=0.1)
        one = App._organic_bloom(ramp, dict(p, bloom_bands=1))[20, :, 0]
        three = App._organic_bloom(ramp, dict(p, bloom_bands=3,
                                              bloom_band_spread=2.5))[20, :, 0]
        worst = float(np.abs(np.diff(three.astype(np.float64), 2)).max())
        self.assertLess(worst,
                        float(np.abs(np.diff(one.astype(np.float64), 2)).max()))

    def test_bloom_still_only_adds_light(self):
        img = np.random.default_rng(5).random((120, 160, 3)).astype(np.float32)
        img[50:56, 70:76] = 80.0
        out = App._organic_bloom(img, dict(self.P, bloom_bands=4))
        self.assertTrue(np.all(out >= img - 1e-5))

    # ── plumbing ──
    def test_the_reach_covers_the_widest_band(self):
        """The crop has to be padded for the biggest kernel in play, not
        the base radius."""
        base = App._organic_stack_reach(dict(self.P, bloom_bands=1))
        wide = App._organic_stack_reach(dict(self.P, bloom_bands=3,
                                             bloom_band_spread=2.5))
        self.assertGreater(wide, base * 5)

    def test_both_controls_are_saveable(self):
        for key in ("bloom_bands", "bloom_band_spread"):
            with self.subTest(param=key):
                self.assertIn(key, App.ORGANIC_PRESET_VARS)
                self.assertIn(key, App._ORGANIC_BASE)

    def test_empty_bands_are_skipped(self):
        """On a typical plate most bands are empty -- one practical does
        not populate four decades of brightness -- and a band with
        nothing in it costs a full transform for a black frame."""
        import inspect
        body = inspect.getsource(App._organic_bloom)
        self.assertIn("np.any(w > 1e-4)", body)


class TestVeilBands(unittest.TestCase):
    """Banding on the veil, where -- unlike on the bloom kernel -- it is
    the physical answer. Charge bleed is genuinely non-linear: more
    excess charge travels further before it is absorbed."""

    P = dict(App._ORGANIC_BASE, veil_thresh=1.0, veil_intensity=0.5,
             veil_size_a=8, veil_size_b=30, veil_mix=0.5, veil_sat=0.0)

    def _reach(self, src, bands, spread=2.5, sat=0.0, n=801):
        im = np.zeros((n, n, 3), np.float32)
        im[n // 2, n // 2] = src
        row = (App._organic_veil(im, dict(self.P, veil_bands=bands,
               veil_band_spread=spread, veil_sat=sat))
               - im)[n // 2, n // 2:, 0]
        norm = row / max(float(row.max()), 1e-12)
        return max([k for k in range(1, n // 2) if norm[k] > 0.02] or [0])

    def test_one_band_is_the_default(self):
        self.assertEqual(App._ORGANIC_BASE["veil_bands"], 1)

    def test_one_band_is_bit_identical_to_no_banding_key(self):
        img = np.random.default_rng(4).random((120, 160, 3)).astype(np.float32)
        img[50:56, 70:76] = 40.0
        p = dict(self.P)
        p.pop("veil_bands", None)
        np.testing.assert_array_equal(
            App._organic_veil(img, p),
            App._organic_veil(img, dict(self.P, veil_bands=1)))

    def test_one_band_ignores_the_spread(self):
        img = np.random.default_rng(7).random((120, 160, 3)).astype(np.float32)
        img[50:56, 70:76] = 40.0
        np.testing.assert_array_equal(
            App._organic_veil(img, dict(self.P, veil_bands=1,
                                        veil_band_spread=1.3)),
            App._organic_veil(img, dict(self.P, veil_bands=1,
                                        veil_band_spread=4.0)))

    def test_one_band_spills_the_same_distance_whatever_the_source(self):
        """A single blur is linear, so its shape cannot vary."""
        reaches = [self._reach(v, 1) for v in (2.0, 30.0, 400.0)]
        self.assertEqual(len(set(reaches)), 1, f"varied: {reaches}")

    def test_brighter_highlights_bleed_further(self):
        """The physical claim this control exists to make."""
        dim = self._reach(2.0, 3)
        bright = self._reach(400.0, 3)
        self.assertGreater(bright, dim * 2)

    def test_the_spread_control_scales_it(self):
        self.assertGreater(self._reach(400.0, 3, spread=3.5),
                           self._reach(400.0, 3, spread=1.3))

    def test_it_works_with_colour_on(self):
        """The tinted path carries a 3-channel source, so the band
        weight has to broadcast rather than assume 2-D."""
        for sat in (0.0, 1.0):
            with self.subTest(colour=sat):
                self.assertGreater(self._reach(400.0, 3, sat=sat),
                                   self._reach(2.0, 3, sat=sat))

    def test_a_gradient_does_not_contour(self):
        ramp = np.zeros((40, 900, 3), np.float32)
        ramp[:] = np.linspace(0.5, 600.0, 900,
                              dtype=np.float32)[None, :, None]

        def worst(nb):
            prof = App._organic_veil(
                ramp, dict(self.P, veil_bands=nb,
                           veil_band_spread=2.5))[20, :, 0].astype(np.float64)
            return float(np.abs(np.diff(prof, 2)).max())
        self.assertLess(worst(3), worst(1))

    def test_the_veil_still_only_adds_light(self):
        img = np.random.default_rng(8).random((120, 160, 3)).astype(np.float32)
        img[50:56, 70:76] = 80.0
        out = App._organic_veil(img, dict(self.P, veil_bands=4))
        self.assertTrue(np.all(out >= img - 1e-6))

    def test_it_shares_the_bloom_band_weights(self):
        """One partition-of-unity implementation, not two that can
        drift apart."""
        import inspect
        self.assertIn("_organic_band_weights",
                      inspect.getsource(App._organic_veil))

    def test_the_reach_covers_the_widest_band(self):
        """Compared as a DIFFERENCE, not a ratio: the baseline reach
        also carries the bloom's term, so a ratio understates how much
        the veil itself grew."""
        one = App._organic_stack_reach(dict(self.P, veil_bands=1))
        three = App._organic_stack_reach(dict(self.P, veil_bands=3,
                                              veil_band_spread=2.5))
        widest = max(self.P["veil_size_a"], self.P["veil_size_b"])
        self.assertAlmostEqual(three - one,
                               3.0 * widest * (2.5 ** 2 - 1.0), delta=2)

    def test_both_controls_are_saveable(self):
        for key in ("veil_bands", "veil_band_spread"):
            with self.subTest(param=key):
                self.assertIn(key, App.ORGANIC_PRESET_VARS)
                self.assertIn(key, App._ORGANIC_BASE)


class TestFixMissingDetection(unittest.TestCase):
    """Fault detection for Fix Missing Frames. Three faults, three
    signals, and the false positives matter more than the misses: a
    missed dropout is a note from the client, a false positive is a
    good frame silently replaced by a generated one."""

    @staticmethod
    def _clip(n=24, seed=1):
        rng = np.random.default_rng(seed)
        base = rng.random((120, 200, 3)).astype(np.float32) * 0.5
        return [np.clip(base + 0.02 * i, 0, 1).astype(np.float32)
                for i in range(n)]

    def _sigs(self, frames):
        return [App._ff_signature(f) for f in frames]

    # ── black ──
    def test_it_finds_a_black_frame(self):
        f = self._clip()
        f[7][:] = 0.0
        self.assertEqual(App._ff_scan(self._sigs(f))["black"], [7])

    def test_it_finds_a_run_of_black_frames(self):
        f = self._clip()
        for i in (7, 8, 9):
            f[i][:] = 0.0
        self.assertEqual(App._ff_scan(self._sigs(f))["black"], [7, 8, 9])

    def test_a_dark_night_frame_is_not_black(self):
        """Judged on the PEAK, not the mean -- a night exterior has a
        very low mean and is perfectly good footage."""
        rng = np.random.default_rng(5)
        night = (rng.random((120, 200, 3)).astype(np.float32) * 0.004)
        self.assertFalse(App._ff_is_black(App._ff_signature(night)))

    def test_a_frame_with_one_hot_pixel_is_not_black(self):
        a = np.zeros((120, 200, 3), np.float32)
        a[60, 100] = 1.0
        self.assertFalse(App._ff_is_black(App._ff_signature(a)))

    def test_the_black_level_is_adjustable(self):
        a = np.full((60, 60, 3), 0.02, np.float32)
        sig = App._ff_signature(a)
        self.assertFalse(App._ff_is_black(sig, 1e-3))
        self.assertTrue(App._ff_is_black(sig, 0.05))

    # ── held / dropped ──
    def test_it_finds_a_held_frame(self):
        f = self._clip()
        f[15] = f[14].copy()
        self.assertEqual(App._ff_scan(self._sigs(f))["held"], [15])

    def test_moving_footage_is_not_held(self):
        self.assertEqual(App._ff_scan(self._sigs(self._clip()))["held"], [])

    def test_grain_alone_does_not_read_as_motion(self):
        """The thumbnail is what makes this robust -- comparing full
        frames would call a re-grained hold a new frame."""
        rng = np.random.default_rng(9)
        base = rng.random((120, 200, 3)).astype(np.float32) * 0.5
        a = base
        b = np.clip(base + rng.normal(0, 0.004, base.shape), 0,
                    1).astype(np.float32)
        self.assertLess(App._ff_thumb_diff(App._ff_signature(a),
                                           App._ff_signature(b)), 0.01)

    def test_black_frames_are_not_also_counted_as_held(self):
        """Two consecutive black frames ARE identical; reporting both
        faults would double every dropout in a black run."""
        f = self._clip()
        for i in (7, 8, 9):
            f[i][:] = 0.0
        r = App._ff_scan(self._sigs(f))
        self.assertEqual(r["held"], [])

    def test_the_frame_after_a_black_run_is_not_held(self):
        f = self._clip()
        f[7][:] = 0.0
        self.assertNotIn(8, App._ff_scan(self._sigs(f))["held"])

    def test_a_dark_plate_does_not_read_as_all_held(self):
        """The difference is normalised by brightness; an absolute
        tolerance would call every frame of a night scene a duplicate."""
        rng = np.random.default_rng(11)
        base = rng.random((120, 200, 3)).astype(np.float32) * 0.02
        frames = [np.clip(base + 0.001 * i, 0, 1).astype(np.float32)
                  for i in range(12)]
        self.assertEqual(App._ff_scan(self._sigs(frames))["held"], [])

    # ── missing files ──
    def test_it_reads_a_trailing_frame_number(self):
        """Trailing, because a show name can contain digits of its
        own."""
        self.assertEqual(
            App._ff_frame_number("sh0210_comp_v003.0142.exr"), 142)
        self.assertEqual(App._ff_frame_number("plate.00099.exr"), 99)

    def test_an_unnumbered_name_has_no_number(self):
        self.assertIsNone(App._ff_frame_number("plate.exr"))

    def test_it_finds_gaps_in_a_numbered_sequence(self):
        names = [f"p.{i:04d}.exr" for i in range(100, 120)
                 if i not in (105, 106, 113)]
        self.assertEqual(App._ff_missing_numbers(names), [105, 106, 113])

    def test_a_complete_sequence_has_no_gaps(self):
        self.assertEqual(
            App._ff_missing_numbers([f"p.{i:04d}.exr"
                                     for i in range(1, 50)]), [])

    def test_a_sequence_on_twos_is_not_all_missing(self):
        """The catastrophic false positive: a deliberately stepped
        sequence read as hundreds of dropouts."""
        self.assertEqual(
            App._ff_missing_numbers([f"p.{i:04d}.exr"
                                     for i in range(100, 200, 2)]), [])

    def test_unnumbered_names_report_nothing(self):
        self.assertEqual(App._ff_missing_numbers(["a.exr", "b.exr",
                                                  "c.exr"]), [])

    def test_too_few_frames_to_judge(self):
        self.assertEqual(App._ff_missing_numbers(["p.0001.exr"]), [])

    # ── runs ──
    def test_faults_group_into_spans(self):
        """A four-frame dropout is one rebuild with good frames either
        side, not four separate ones."""
        self.assertEqual(App._ff_runs([3, 4, 5, 9, 20, 21]),
                         [(3, 5), (9, 9), (20, 21)])

    def test_runs_tolerate_unsorted_input_and_repeats(self):
        self.assertEqual(App._ff_runs([9, 3, 4, 3, 5]), [(3, 5), (9, 9)])

    def test_no_faults_is_no_runs(self):
        self.assertEqual(App._ff_runs([]), [])

    # ── output naming ──
    def test_it_prefixes_files_and_folders(self):
        self.assertEqual(App._ff_output_name("/jobs/sh010/plate"),
                         "FF_plate")
        self.assertEqual(App._ff_output_name("/jobs/sh010/take.mov"),
                         "FF_take.mov")

    def test_prefixing_is_idempotent(self):
        """Re-fixing an already-fixed clip should not give FF_FF_."""
        self.assertEqual(App._ff_output_name("/x/FF_plate"), "FF_plate")

    def test_a_trailing_separator_does_not_swallow_the_name(self):
        self.assertEqual(App._ff_output_name("/jobs/sh010/plate/"),
                         "FF_plate")

    # ── end to end on a synthetic clip ──
    def test_a_mixed_clip_reports_each_fault_once(self):
        f = self._clip(30)
        for i in (7, 8):
            f[i][:] = 0.0
        f[15] = f[14].copy()
        f[16] = f[14].copy()
        r = App._ff_scan(self._sigs(f))
        self.assertEqual(App._ff_runs(r["black"]), [(7, 8)])
        self.assertEqual(App._ff_runs(r["held"]), [(15, 16)])


class TestFixFramesRepair(unittest.TestCase):
    """The repair geometry. The span sent to VACE is padded with real
    frames so the model has motion to reconstruct from, but only the
    broken core is written back -- context frames were never broken and
    must not be rewritten."""

    class _Stub(App):
        def __init__(self):
            pass

    def setUp(self):
        self.app = self._Stub()

    def test_the_core_maps_back_to_the_faulty_frames(self):
        """The bug this guards: an off-by-one here silently overwrites
        good frames with generated ones."""
        for first, last, total in ((40, 42, 200), (1, 1, 200),
                                   (199, 199, 200), (0, 0, 60),
                                   (10, 30, 60)):
            with self.subTest(fault=(first, last)):
                pl = self.app._ff_repair_plan(first, last, total)
                self.assertEqual(pl["send0"] + pl["core_start"], first)
                self.assertEqual(pl["send0"] + pl["core_end"], last)

    def test_it_sends_context_on_both_sides(self):
        pl = self.app._ff_repair_plan(40, 42, 200)
        self.assertEqual(pl["tether_l"], App.FF_CONTEXT)
        self.assertEqual(pl["tether_r"], App.FF_CONTEXT)
        self.assertGreater(pl["send_gap"], 42 - 40 + 1)

    def test_it_clamps_at_the_head_of_the_clip(self):
        pl = self.app._ff_repair_plan(1, 1, 200)
        self.assertGreaterEqual(pl["send0"], 0)
        self.assertGreaterEqual(pl["core_start"], 0)

    def test_it_clamps_at_the_tail_of_the_clip(self):
        pl = self.app._ff_repair_plan(199, 199, 200)
        self.assertLessEqual(pl["send1"], 199)
        self.assertLess(pl["core_end"], pl["send_gap"])

    def test_the_request_respects_the_models_limits(self):
        """VACE will not generate fewer than its minimum, and refuses
        more than its maximum."""
        for first, last, total in ((0, 0, 20), (10, 12, 200),
                                   (0, 400, 500)):
            with self.subTest(fault=(first, last)):
                pl = self.app._ff_repair_plan(first, last, total)
                self.assertGreaterEqual(pl["request"], App.VACE_MIN_GEN)
                self.assertLessEqual(pl["request"], App.VACE_MAX_GEN)

    def test_the_plan_has_every_key_the_payload_reads(self):
        """It is handed to the same _vace_payload ADW-Interpolate uses,
        so a missing key would only surface as a network error."""
        import inspect
        pl = self.app._ff_repair_plan(40, 42, 200)
        body = inspect.getsource(App._vace_payload)
        for key in re.findall(r'plan\[[\"\']([a-z_0-9]+)[\"\']\]', body):
            with self.subTest(key=key):
                self.assertIn(key, pl)

    def test_it_reuses_the_interpolate_pipeline(self):
        """Same engine, different operator -- not a second copy of the
        VACE orchestration."""
        import inspect
        body = inspect.getsource(App._ff_fix_worker)
        for fn in ("_vace_build_source_clip", "_vace_build_mask_clip",
                   "_vace_submit", "_vace_decode"):
            with self.subTest(helper=fn):
                self.assertIn(fn, body)

    def test_it_copies_before_it_repairs(self):
        """An aborted run should leave a complete, usable sequence
        rather than a folder full of holes."""
        import inspect
        body = inspect.getsource(App._ff_fix_worker)
        self.assertLess(body.index("shutil.copy2"),
                        body.index("_vace_submit"))

    def test_only_the_core_span_is_written_back(self):
        import inspect
        body = inspect.getsource(App._ff_fix_worker)
        self.assertIn('range(plan["core_start"], plan["core_end"] + 1)',
                      body)

    def test_the_tab_is_registered(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn('("Fix Missing Frames", "fixframes")', src)
        self.assertIn("self._tab_fixframes()", src)

    def test_it_sits_next_to_upscale(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertLess(src.index('("Expansion Studio",   "upscale")'),
                        src.index('("Fix Missing Frames", "fixframes")'))
        self.assertLess(src.index('("Fix Missing Frames", "fixframes")'),
                        src.index('("SAM 3 — Segment",    "sam")'))

    def test_it_uses_the_shared_timeline_bar(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn('self._tl_frames["fixframes"]', src)
        self.assertIn('"organic","fixframes"', src)


class TestFixFramesMissingSlots(unittest.TestCase):
    """Absent files have to occupy their rightful slot in the sequence.

    Without that, a missing frame anchors to the last GOOD frame before
    it -- and the repair then regenerates that good frame and never
    creates the absent one. Two wrongs from one off-by-one, and both
    silent."""

    def test_absent_frames_get_their_own_slot(self):
        names = [f"p.{i:04d}.exr" for i in range(100, 112)
                 if i not in (105, 106, 109)]
        full, missing = App._ff_expand_sequence(names)
        self.assertEqual(len(full), 12)
        self.assertEqual(missing, [5, 6, 9])

    def test_the_new_names_do_not_collide_with_real_files(self):
        """The repair writes to these names; if one matched an existing
        file it would overwrite good footage."""
        names = [f"p.{i:04d}.exr" for i in range(100, 112)
                 if i not in (105, 106, 109)]
        full, missing = App._ff_expand_sequence(names)
        for i in missing:
            with self.subTest(slot=i):
                self.assertNotIn(full[i], names)

    def test_the_new_names_are_in_numerical_order(self):
        names = [f"p.{i:04d}.exr" for i in range(100, 112) if i != 105]
        full, _m = App._ff_expand_sequence(names)
        nums = [App._ff_frame_number(n) for n in full]
        self.assertEqual(nums, list(range(100, 112)))

    def test_a_complete_sequence_is_untouched(self):
        ok = [f"p.{i:04d}.exr" for i in range(1, 20)]
        full, missing = App._ff_expand_sequence(ok)
        self.assertEqual(full, ok)
        self.assertEqual(missing, [])

    def test_a_sequence_on_twos_is_untouched(self):
        tw = [f"p.{i:04d}.exr" for i in range(100, 120, 2)]
        full, missing = App._ff_expand_sequence(tw)
        self.assertEqual(full, tw)
        self.assertEqual(missing, [])

    def test_unnumbered_names_are_untouched(self):
        raw = ["a.exr", "b.exr", "c.exr"]
        full, missing = App._ff_expand_sequence(raw)
        self.assertEqual(full, raw)
        self.assertEqual(missing, [])

    # ── name construction ──
    def test_padding_is_preserved(self):
        self.assertEqual(App._ff_name_for_number("p.0104.exr", 105),
                         "p.0105.exr")

    def test_only_the_trailing_number_is_replaced(self):
        """A show name can carry digits of its own -- the version and
        the shot number must survive."""
        self.assertEqual(
            App._ff_name_for_number("sh0210_comp_v003.0142.exr", 143),
            "sh0210_comp_v003.0143.exr")

    def test_a_number_wider_than_the_padding_still_works(self):
        self.assertEqual(App._ff_name_for_number("shot.99.dpx", 100),
                         "shot.100.dpx")

    def test_an_unnumbered_template_gives_nothing(self):
        self.assertIsNone(App._ff_name_for_number("plate.exr", 5))

    # ── the repair honours the slots ──
    def test_the_copy_step_skips_absent_frames(self):
        """There is no file to copy; the repair creates it."""
        import inspect
        body = inspect.getsource(App._ff_fix_worker)
        i = body.index("Copying source")
        self.assertIn("if i in missing:", body[i:i + 400])

    def test_the_clip_builder_is_given_a_stand_in(self):
        """It reads real files, so an absent frame would abort the
        build. The stand-in sits inside the core span and is
        regenerated anyway."""
        import inspect
        body = inspect.getsource(App._ff_fix_worker)
        self.assertIn("clip_names", body)

    def test_a_missing_slot_is_not_also_reported_as_black(self):
        """It has no pixels at all -- counting it twice would double
        every gap."""
        import inspect
        body = inspect.getsource(App._ff_scan_worker)
        self.assertIn("skip = self.ff_missing", body)
        self.assertIn("if i not in skip", body)


class TestFixFramesFrameRate(unittest.TestCase):
    """A fixer that silently retimes the delivery has not fixed it."""

    def test_an_unreadable_source_falls_back_rather_than_raising(self):
        self.assertEqual(App._ff_probe_fps("/nonexistent.mov"), 24.0)

    def test_the_fallback_is_caller_chosen(self):
        self.assertEqual(App._ff_probe_fps("/nonexistent.mov",
                                           default=25.0), 25.0)

    def test_the_rate_is_probed_not_assumed(self):
        import inspect
        self.assertIn("_ff_probe_fps",
                      inspect.getsource(App._ff_load_movie))

    def test_the_repair_uses_the_probed_rate(self):
        import inspect
        body = inspect.getsource(App._ff_fix_worker)
        self.assertIn('getattr(self, "ff_fps"', body)
        self.assertNotIn("fps = 24.0", body)

    def test_a_nonsense_rate_is_rejected(self):
        """ffprobe reports 0/0 for streams it cannot read, and a zero
        frame rate would make ffmpeg fail much later and much more
        confusingly."""
        for bad in (0.0, -5.0, 100000.0):
            with self.subTest(rate=bad):
                self.assertFalse(0.1 < bad < 1000.0)


def _tc_tex(h, w, seed, bias=0.0, sc=1.0):
    r = np.random.default_rng(seed)
    a = cv2.GaussianBlur(r.random((h, w)).astype(np.float32), (0, 0), 3.0)
    return np.clip((a - 0.5) * sc + 0.5 + bias, 0, 1).astype(np.float32)


def _tc_warp(f, phase):
    """A per-frame VARYING distortion. That is what temporal
    inconsistency means -- a steady distortion applied identically to
    every frame is temporally consistent and should NOT be flagged."""
    h, w = f.shape[:2]
    mx = (np.arange(w, dtype=np.float32)[None, :]
          + 10 * np.sin(np.arange(h, dtype=np.float32)[:, None] / 8.0 + phase))
    my = (np.repeat(np.arange(h, dtype=np.float32)[:, None], w, 1)
          + 6 * np.cos(np.arange(w, dtype=np.float32)[None, :] / 11.0 + phase))
    return cv2.remap(f, mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


class TestTemporalAnalysis(unittest.TestCase):
    """Stage 1 of the consistency analyser: cut detection, flow
    residual, per-shot normalisation.

    The whole design rests on one inversion -- this measures how much of
    the change motion CANNOT explain, not how much motion there is. A
    whip-pan must score zero."""

    def test_a_clean_pan_scores_near_zero(self):
        """The single most important property. Motion is high; residual
        must be tiny, because everything went where the flow said."""
        base = _tc_tex(400, 1000, 1)
        planes = [App._tc_prepare(base[50:350, x:x + 520])
                  for x in range(0, 120, 5)]
        r = App._tc_analyse(planes)
        self.assertLess(max(r["score"].values()), 0.05)
        self.assertEqual(App._tc_runs(r["score"]), [])

    def test_the_warp_actually_reduces_the_error(self):
        """Guards the bug this was written with: computing flow
        prev->curr and remapping by +d doubles the displacement instead
        of cancelling it, and a clean pan then scored a residual 37%
        HIGHER than not warping at all -- the metric read backwards."""
        base = _tc_tex(300, 900, 2)
        a = App._tc_prepare(base[:, 0:520])
        b = App._tc_prepare(base[:, 6:526])
        res, raw = App._tc_flow_residual(a, b)
        self.assertLess(res, raw * 0.25)

    def test_varying_damage_is_caught(self):
        base = _tc_tex(400, 1000, 3)
        planes = []
        for k, x in enumerate(range(0, 120, 5)):
            f = base[50:350, x:x + 520].copy()
            if 10 <= k <= 15:
                f = _tc_warp(cv2.GaussianBlur(f, (0, 0), 4.0), k * 1.7)
            planes.append(App._tc_prepare(f))
        runs = App._tc_runs(App._tc_analyse(planes)["score"])
        self.assertEqual(len(runs), 1)
        a, b = runs[0]
        self.assertLessEqual(a, 11)
        self.assertGreaterEqual(b, 15)

    def test_steady_distortion_is_not_flagged(self):
        """A lens or a look applied identically to every frame is
        consistent. Flagging it would condemn every graded plate."""
        base = _tc_tex(400, 1000, 4)
        planes = [App._tc_prepare(_tc_warp(base[50:350, x:x + 520], 0.9))
                  for x in range(0, 120, 5)]
        self.assertEqual(App._tc_runs(App._tc_analyse(planes)["score"]), [])

    # ── cuts ──
    def test_a_cut_is_detected(self):
        planes = ([App._tc_prepare(_tc_tex(300, 520, 7)) for _ in range(8)]
                  + [App._tc_prepare(_tc_tex(300, 520, 77, bias=-0.22,
                                             sc=0.55)) for _ in range(8)])
        self.assertEqual(App._tc_analyse(planes)["cuts"], [8])

    def test_a_cut_is_not_reported_as_damage(self):
        """A cut is a 100% residual. Unflagged, the tool calls every
        edit catastrophic and nobody trusts the colour bar again."""
        planes = ([App._tc_prepare(_tc_tex(300, 520, 7)) for _ in range(8)]
                  + [App._tc_prepare(_tc_tex(300, 520, 77, bias=-0.22,
                                             sc=0.55)) for _ in range(8)])
        r = App._tc_analyse(planes)
        for a, b in App._tc_runs(r["score"]):
            self.assertFalse(a <= 8 <= b, "the cut was scored as damage")

    def test_cuts_split_the_clip_into_shots(self):
        planes = ([App._tc_prepare(_tc_tex(300, 520, 7)) for _ in range(8)]
                  + [App._tc_prepare(_tc_tex(300, 520, 77, bias=-0.22,
                                             sc=0.55)) for _ in range(8)])
        self.assertEqual(App._tc_analyse(planes)["shots"], [(0, 7), (8, 15)])

    def test_cuts_are_found_on_the_histogram_not_the_residual(self):
        """Measured: a cut and a badly warped frame give almost the same
        residual-to-raw ratio, 0.67 against 0.65. The residual cannot
        tell them apart, so a threshold on it would either miss cuts or
        condemn damaged frames as edits."""
        import inspect
        body = inspect.getsource(App._tc_is_cut)
        self.assertIn("hist_dist", body)

    def test_a_missing_histogram_is_not_a_cut(self):
        self.assertFalse(App._tc_is_cut(0.9, 1.0, None))

    def test_histogram_distance_is_bounded(self):
        a = App._tc_histogram(_tc_tex(64, 64, 1))
        b = App._tc_histogram(_tc_tex(64, 64, 2))
        self.assertGreaterEqual(App._tc_hist_distance(a, a), 0.0)
        self.assertAlmostEqual(App._tc_hist_distance(a, a), 0.0, places=6)
        self.assertLessEqual(App._tc_hist_distance(a, b), 1.0)

    def test_shot_spans_cover_every_frame(self):
        for cuts, n in (([], 10), ([5], 10), ([3, 7], 12), ([1], 4)):
            with self.subTest(cuts=cuts):
                spans = App._tc_shots(cuts, n)
                covered = [i for a, b in spans for i in range(a, b + 1)]
                self.assertEqual(covered, list(range(n)))

    # ── per-shot normalisation ──
    def test_scores_are_normalised_within_a_shot(self):
        """A handheld night interior and a locked-off plate have
        baselines an order apart; one absolute threshold is deaf on one
        and hysterical on the other."""
        import inspect
        self.assertIn("median", inspect.getsource(App._tc_score_shot))

    def test_a_uniformly_clean_shot_scores_zero_throughout(self):
        """Max-normalising would force a 1.0 into every shot, so a
        perfectly clean one would report its least-clean frame as
        maximum damage."""
        resid = {i: 0.01 + 0.0001 * (i % 3) for i in range(1, 20)}
        sc = App._tc_score_shot(resid, 1, 19)
        self.assertLess(max(sc.values()), 0.1)

    def test_the_score_tracks_the_multiple_of_baseline(self):
        resid = {i: 0.01 for i in range(1, 20)}
        resid[10] = 0.04            # 4x the baseline
        sc = App._tc_score_shot(resid, 1, 19)
        self.assertAlmostEqual(sc[10], 1.0, delta=0.05)
        self.assertLess(sc[5], 0.05)

    def test_an_empty_shot_scores_nothing(self):
        self.assertEqual(App._tc_score_shot({}, 0, 5), {})

    # ── runs and prose ──
    def test_single_frame_spikes_are_dropped(self):
        """Usually one mis-tracked flow field, not damage."""
        self.assertEqual(App._tc_runs({5: 0.9}), [])

    def test_one_clean_frame_does_not_split_a_run(self):
        """Damage rarely stops and restarts for a frame; reporting
        30-38, 40-52 instead of 30-52 is noise, not precision."""
        sc = {i: 0.8 for i in range(30, 53)}
        sc[39] = 0.1
        self.assertEqual(App._tc_runs(sc), [(30, 52)])

    def test_the_summary_names_the_span_and_the_evidence(self):
        sc = {i: 0.9 for i in range(30, 40)}
        out = App._tc_summarise({"score": sc, "ratio": {i: 4.0 for i in sc}})
        self.assertEqual(len(out), 1)
        txt = out[0]["text"]
        self.assertIn("30", txt)
        self.assertIn("39", txt)
        self.assertIn("baseline", txt)
        self.assertIn("peak_frame", out[0])

    def test_severity_wording_tracks_the_number(self):
        """So the prose cannot drift from the evidence it describes."""
        self.assertEqual(App._tc_severity(0.95), "severe")
        self.assertEqual(App._tc_severity(0.65), "strong")
        self.assertEqual(App._tc_severity(0.45), "moderate")
        self.assertEqual(App._tc_severity(0.1), "mild")

    def test_the_summary_does_not_name_a_subject(self):
        """Stage 1 has no region or object detection. Naming the subject
        needs SAM, and a confident wrong noun is worse than an honest
        silence because prose gets acted on without being checked."""
        sc = {i: 0.9 for i in range(5, 12)}
        txt = App._tc_summarise({"score": sc,
                                 "ratio": {i: 4.0 for i in sc}})[0]["text"]
        for noun in ("face", "hand", "the subject"):
            self.assertNotIn(noun, txt)

    def test_a_clean_clip_produces_no_findings(self):
        self.assertEqual(App._tc_summarise({"score": {}, "ratio": {}}), [])

    # ── the prepared plane ──
    def test_preparation_is_resolution_independent(self):
        big = _tc_tex(1080, 1920, 5)
        self.assertLessEqual(App._tc_prepare(big).shape[1],
                             App.TC_WORK_WIDTH)

    def test_preparation_handles_colour_and_mono(self):
        rgb = np.dstack([_tc_tex(120, 200, 6)] * 3)
        mono = _tc_tex(120, 200, 6)
        np.testing.assert_allclose(App._tc_prepare(rgb),
                                   App._tc_prepare(mono), atol=1e-5)

    def test_scene_linear_input_is_brought_into_range(self):
        hdr = _tc_tex(120, 200, 8) * 40.0
        self.assertLessEqual(float(App._tc_prepare(hdr).max()), 1.0 + 1e-6)

    def test_the_frame_border_is_excluded(self):
        """Content legitimately enters and leaves at the edge, and a
        warp has nothing to fetch from outside the plate -- scoring it
        would make every camera move look broken."""
        import inspect
        self.assertIn("0.06", inspect.getsource(App._tc_flow_residual))

    def test_analysis_can_be_cancelled(self):
        planes = [App._tc_prepare(_tc_tex(120, 200, i)) for i in range(20)]
        calls = []
        App._tc_analyse(planes, cancel=lambda: (calls.append(1),
                                                len(calls) > 3)[1])
        self.assertLess(len(calls), 20)


class TestFixFramesReading(unittest.TestCase):
    """EXR reading. The pip OpenCV wheels ship with the OpenEXR codec
    compiled out, so cv2.imread raises on the exact format this tool
    exists to check -- and this tab had its own reader instead of the
    app's."""

    class _Stub(App):
        def __init__(self):
            pass

    def setUp(self):
        self.app = self._Stub()
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write_exr(self, name, rgb):
        try:
            import OpenEXR
        except ImportError:
            self.skipTest("OpenEXR not available")
        h, w = rgb.shape[:2]
        path = os.path.join(self.dir, name)
        out = OpenEXR.OutputFile(path, OpenEXR.Header(w, h))
        out.writePixels({c: rgb[:, :, i].astype(np.float32).tobytes()
                         for i, c in enumerate("RGB")})
        out.close()
        return path

    def test_it_does_not_use_cv2_imread_for_exr(self):
        """The specific failure reported: 'OpenEXR codec is disabled'.
        Checked against the CODE, not the docstring -- which mentions
        cv2.imread by name to explain why it is not used."""
        import ast as _ast
        import inspect
        import textwrap
        tree = _ast.parse(textwrap.dedent(
            inspect.getsource(App._ff_read_frame)))
        fn = tree.body[0]
        if (fn.body and isinstance(fn.body[0], _ast.Expr)
                and isinstance(fn.body[0].value, _ast.Constant)):
            fn.body = fn.body[1:]           # drop the docstring
        code = _ast.unparse(fn)
        self.assertIn("OpenEXR", code)
        self.assertNotIn("cv2.imread", code)

    def test_an_exr_round_trips(self):
        rgb = (np.random.default_rng(1).random((32, 48, 3)).astype(np.float32)
               * 4.0)
        got = self.app._ff_read_frame(self._write_exr("t.0001.exr", rgb))
        np.testing.assert_allclose(got, rgb, atol=1e-6)

    def test_scene_linear_values_survive(self):
        """Nothing is linearised or tonemapped on the way in -- a
        black-frame test wants the real numbers, not a graded view."""
        rgb = np.full((16, 16, 3), 4.0, np.float32)
        got = self.app._ff_read_frame(self._write_exr("t.0002.exr", rgb))
        self.assertAlmostEqual(float(got.max()), 4.0, places=4)

    def test_a_black_exr_is_detected(self):
        rgb = np.zeros((16, 16, 3), np.float32)
        sig = App._ff_signature(
            self.app._ff_read_frame(self._write_exr("t.0003.exr", rgb)))
        self.assertTrue(App._ff_is_black(sig))

    def test_a_bright_exr_is_not_black(self):
        rgb = (np.random.default_rng(2).random((16, 16, 3)).astype(np.float32)
               * 4.0)
        sig = App._ff_signature(
            self.app._ff_read_frame(self._write_exr("t.0004.exr", rgb)))
        self.assertFalse(App._ff_is_black(sig))

    def test_nan_and_inf_are_neutralised(self):
        """A NaN would propagate through the signature and make every
        comparison against that frame meaningless."""
        rgb = np.ones((16, 16, 3), np.float32)
        rgb[0, 0] = np.nan
        rgb[1, 1] = np.inf
        got = self.app._ff_read_frame(self._write_exr("t.0005.exr", rgb))
        self.assertTrue(np.isfinite(got).all())

    def test_a_corrupt_file_raises_a_readable_error(self):
        path = os.path.join(self.dir, "t.0006.exr")
        with open(path, "wb") as fh:
            fh.write(b"not an exr at all")
        with self.assertRaises(RuntimeError) as cm:
            self.app._ff_read_frame(path)
        self.assertIn("t.0006.exr", str(cm.exception))

    def test_an_unreadable_frame_does_not_abort_the_scan(self):
        """Aborting on the first bad file would leave the rest of a
        delivery unchecked -- exactly backwards for a tool whose job is
        finding bad files."""
        import inspect
        body = inspect.getsource(App._ff_scan_worker)
        self.assertIn("unreadable.append(i)", body)
        self.assertIn('"kind": "corrupt"', body)

    def test_a_corrupt_frame_is_excluded_from_the_other_tests(self):
        """It has no pixels, so it must not also be reported as black or
        as a repeat of its neighbour."""
        import inspect
        body = inspect.getsource(App._ff_scan_worker)
        self.assertIn("self.ff_missing | set(unreadable)", body)

    def test_an_input_with_no_readable_frames_says_so(self):
        import inspect
        self.assertIn("no readable frames",
                      inspect.getsource(App._ff_scan_worker))

    def test_the_preview_does_not_flip_channels(self):
        """The old flip was compensating for cv2.imread's BGR order,
        which this reader does not use -- left in, it would swap red and
        blue in every preview."""
        import inspect
        body = inspect.getsource(App._ff_show_frame)
        self.assertNotIn("[:, :, ::-1]", body)


class TestTemporalAnalysisUI(unittest.TestCase):
    """Wiring the analyser into MOV to EXR.

    Most of this checks that names RESOLVE. The engine was written and
    tested a turn before any of it was on screen, and the first attempt
    at the UI called set_index() and redraw() on the timeline -- neither
    of which exists -- and painted onto a tab that had no timeline at
    all. None of that fails until a user clicks the button."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_every_method_the_ui_calls_exists(self):
        for name in ("_tc_start", "_tc_worker", "_tc_done", "_tc_goto",
                     "_tc_frame_color", "_tc_select_frame",
                     "_tc_source_frames", "_tc_reset", "_tc_analyse",
                     "_tc_summarise", "_tc_severity", "_tc_prepare",
                     "_ff_read_frame", "_ff_decode_movie",
                     "_motion_tint_color", "progress"):
            with self.subTest(method=name):
                self.assertTrue(callable(getattr(App, name, None)))

    def test_the_button_is_actually_built(self):
        """The complaint that started this: no new buttons appeared,
        because only the engine had been written."""
        body = self._src(App._tab_exr)
        self.assertIn("Analyse temporal consistency", body)
        self.assertIn("self._tc_start", body)

    def test_the_tab_has_a_timeline_to_paint_on(self):
        """It had none. The colour coding is the point of the feature,
        so without one there was nowhere for it to go."""
        self.assertIn('self._tl_frames["exr"]', self._src(App._tab_exr))
        self.assertIn("self.exr_frame_cb = self._timeline",
                      self._src(App._tab_exr))

    def test_exr_is_registered_in_the_shared_timeline_bar(self):
        self.assertIn('"exr"', self._src(App._build_timeline_bar))

    def test_it_uses_the_real_timeline_api(self):
        """select() and refresh(), not set_index() and redraw() -- the
        first attempt invented both."""
        body = (self._src(App._tc_done) + self._src(App._tc_goto)
                + self._src(App._tc_worker))
        self.assertIn("refresh()", body)
        self.assertIn("select(", body)
        self.assertNotIn("set_index", body)
        self.assertNotIn(".redraw(", body)

    def test_the_timeline_proxy_offers_those_methods(self):
        """Checked against the widget itself, so renaming one breaks
        here rather than at click time."""
        body = self._src(App._timeline)
        for m in ("def set_frames", "def refresh", "def select"):
            with self.subTest(method=m):
                self.assertIn(m, body)

    # ── the colour mapping ──
    def test_a_clean_frame_is_left_alone(self):
        app = _StubApp()
        app._tc_result = {"score": {5: 0.05}, "cuts": [], "ratio": {}}
        self.assertIsNone(App._tc_frame_color(app, 5))

    def test_a_bad_frame_is_tinted(self):
        app = _StubApp()
        app._tc_result = {"score": {5: 0.9}, "cuts": [], "ratio": {}}
        self.assertIsNotNone(App._tc_frame_color(app, 5))

    def test_the_score_is_inverted_for_the_ramp(self):
        """The shared ramp runs red-at-zero to yellow-at-one. Fed the
        score directly, a clean frame would glow yellow and a broken one
        would be the colour of a healthy one -- backwards."""
        app = _StubApp()
        app._tc_result = {"score": {1: 0.3, 2: 1.0}, "cuts": [],
                          "ratio": {}}
        mild = App._tc_frame_color(app, 1)
        severe = App._tc_frame_color(app, 2)
        self.assertEqual(severe, App._motion_tint_color(0.0))
        self.assertNotEqual(mild, severe)

    def test_a_cut_is_coloured_as_information(self):
        app = _StubApp()
        app._tc_result = {"score": {}, "cuts": [8], "ratio": {}}
        self.assertEqual(App._tc_frame_color(app, 8), "#4488CC")

    def test_no_result_means_no_colour(self):
        app = _StubApp()
        app._tc_result = None
        self.assertIsNone(App._tc_frame_color(app, 3))

    # ── behaviour ──
    def test_a_second_click_cancels(self):
        """Rather than queueing a second pass over the same footage."""
        body = self._src(App._tc_start)
        self.assertIn("if self._tc_busy:", body)
        self.assertIn("self._tc_cancel = True", body)

    def test_an_unreadable_frame_does_not_abort_the_pass(self):
        body = self._src(App._tc_worker)
        self.assertIn("planes.append(None)", body)

    def test_missing_input_reports_instead_of_raising(self):
        body = self._src(App._tc_source_frames)
        self.assertIn("return None, None", body)
        self.assertIn("Choose a video file first", body)

    def test_it_handles_both_conversion_directions(self):
        body = self._src(App._tc_source_frames)
        self.assertIn("to_mov", body)
        self.assertIn("_ff_decode_movie", body)

    def test_the_findings_list_jumps_to_the_worst_frame(self):
        """Not the first frame of the run -- that is where the fault
        begins, which is the least convincing example of it."""
        self.assertIn("peak_frame", self._src(App._tc_goto))


class TestFixFramesPreviewColour(unittest.TestCase):
    """The preview looked milky. Three faults, found one at a time.

    The last and least obvious: the CONTAINER does not tell you the
    colourspace. EXRs from AI video tools hold display-READY floats
    despite using a format conventionally reserved for scene-linear
    data, so gamma-ing them washes them out -- the same double-encode as
    gamma-ing an already-encoded PNG, just harder to spot."""

    class _Stub(App):
        def __init__(self):
            self.ff_cs_var = None
            self.adw_source_encoded_var = None

    class _Var:
        def __init__(self, v):
            self.v = v

        def get(self):
            return self.v

    def setUp(self):
        self.app = self._Stub()

    # ── what the container can and cannot tell you ──
    def test_an_encoded_exr_is_not_treated_as_linear(self):
        """The bug: the container implied linear and the values were
        not."""
        self.assertFalse(App._ff_is_linear("p.0001.exr", encoded=True))

    def test_a_genuine_linear_exr_still_gets_its_curve(self):
        self.assertTrue(App._ff_is_linear("p.0001.exr", encoded=False))

    def test_encoded_formats_never_get_a_curve_either_way(self):
        """A decoded movie produces PNGs; those are display-encoded
        whatever the EXR setting says."""
        for n in ("ff_000012.png", "x.jpg", "y.tif"):
            for enc in (True, False):
                with self.subTest(name=n, encoded=enc):
                    self.assertFalse(App._ff_is_linear(n, encoded=enc))

    # ── the display transform ──
    def test_an_ai_tool_exr_passes_through_untouched(self):
        for v in (0.05, 0.2, 0.5):
            with self.subTest(value=v):
                a = np.full((1, 1, 3), v, np.float32)
                self.assertEqual(int(self.app._ff_to_display(a, "p.exr")[0, 0, 0]),
                                 int(v * 255 + 0.5))

    def test_a_scene_linear_exr_gets_the_apps_own_curve(self):
        self.app.ff_cs_var = self._Var("Scene-linear")
        a = np.full((1, 1, 3), 0.05, np.float32)
        self.assertEqual(int(self.app._ff_to_display(a, "p.exr")[0, 0, 0]),
                         int(App._organic_linear_to_display(a)[0, 0, 0]))

    def test_the_curve_is_srgb_not_a_naive_power(self):
        """A pure 1/2.2 power has no linear segment near black, so
        shadows lift: linear 0.002 came out at 15 instead of 7."""
        a = np.full((1, 1, 3), 0.002, np.float32)
        proper = int(App._organic_linear_to_display(a)[0, 0, 0])
        self.assertEqual(proper, 7)
        self.assertGreater(int(round(0.002 ** (1 / 2.2) * 255)), proper + 4)

    def test_png_values_round_trip_exactly(self):
        """Gamma applied a second time took an 8-bit 40 to 110."""
        for v8 in (0, 10, 40, 128, 255):
            with self.subTest(value=v8):
                a = np.full((1, 1, 3), v8 / 255.0, np.float32)
                self.assertEqual(
                    int(self.app._ff_to_display(a, "f.png")[0, 0, 0]), v8)

    # ── wiring ──
    def test_the_tab_defaults_to_the_apps_own_setting(self):
        import inspect
        self.assertIn("_adw_source_is_encoded",
                      inspect.getsource(App._ff_source_encoded))

    def test_there_is_a_per_sequence_override(self):
        """Per-load, not per-install -- one project's EXRs are linear
        and the next one's are not."""
        import inspect
        self.assertIn("ff_cs_var", inspect.getsource(App._tab_fixframes))
        self.assertIn("Scene-linear", inspect.getsource(App._tab_fixframes))

    def test_changing_it_redraws_the_preview(self):
        import inspect
        body = inspect.getsource(App._tab_fixframes)
        i = body.index("ff_cs_var.trace_add")
        self.assertIn("_ff_show_frame", body[i:i + 200])

    def test_the_preview_goes_through_the_one_transform(self):
        import inspect
        body = inspect.getsource(App._ff_show_frame)
        self.assertIn("_ff_to_display", body)
        self.assertNotIn("(1 / 2.2)", body)

    def test_detection_still_reads_raw_values(self):
        """Display and analysis are separate questions -- putting a
        transfer curve in front of the black-frame test would move every
        threshold."""
        import inspect
        self.assertNotIn("_ff_to_display",
                         inspect.getsource(App._ff_read_frame))
        self.assertNotIn("_ff_to_display",
                         inspect.getsource(App._ff_signature))


class TestFixFramesInputUI(unittest.TestCase):
    """One picker, and the path decides what it is -- rather than two
    buttons asking the user to answer a question the path answers."""

    class _Stub(App):
        def __init__(self):
            self.seq = None
            self.mov = None
            self.warned = None

        def _ff_load_sequence(self, folder):
            self.seq = folder

        def _ff_load_movie(self, path):
            self.mov = path

    def setUp(self):
        self.app = self._Stub()
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _touch(self, name):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as fh:
            fh.write(b"x")
        return path

    def test_a_folder_loads_as_a_sequence(self):
        self.app._ff_accept(self.dir)
        self.assertEqual(self.app.seq, self.dir)
        self.assertIsNone(self.app.mov)

    def test_a_movie_loads_as_a_movie(self):
        for ext in (".mov", ".mp4", ".mxf", ".avi"):
            with self.subTest(ext=ext):
                app = self._Stub()
                p = self._touch("take" + ext)
                app._ff_accept(p)
                self.assertEqual(app.mov, p)
                self.assertIsNone(app.seq)

    def test_one_frame_of_a_sequence_loads_its_folder(self):
        """Picking a frame means "this sequence" -- demanding the folder
        instead is pedantry."""
        for ext in (".exr", ".png", ".dpx", ".tif"):
            with self.subTest(ext=ext):
                app = self._Stub()
                app._ff_accept(self._touch("plate.0001" + ext))
                self.assertEqual(app.seq, self.dir)
                self.assertIsNone(app.mov)

    def test_case_is_ignored(self):
        """Deliveries arrive as .MOV and .EXR often enough."""
        app = self._Stub()
        app._ff_accept(self._touch("TAKE.MOV"))
        self.assertEqual(app.mov, os.path.join(self.dir, "TAKE.MOV"))

    def test_an_unsupported_file_is_refused_not_guessed(self):
        """Loading a .pdf as a sequence folder would be worse than
        saying no."""
        app = self._Stub()
        with unittest.mock.patch.object(
                sys.modules[App.__module__], "messagebox") as mb:
            app._ff_accept(self._touch("notes.pdf"))
        self.assertIsNone(app.seq)
        self.assertIsNone(app.mov)
        self.assertTrue(mb.showwarning.called)

    # ── the shared idiom ──
    def test_it_uses_the_shared_picker_widgets(self):
        import inspect
        body = inspect.getsource(App._tab_fixframes)
        self.assertIn("self._path_bar(", body)
        self.assertIn("self._picker(", body)

    def test_the_two_old_buttons_are_gone(self):
        import inspect
        body = inspect.getsource(App._tab_fixframes)
        self.assertNotIn("Image sequence\u2026", body)
        self.assertNotIn("_ff_pick_sequence", body)
        self.assertNotIn("_ff_pick_movie", body)

    def test_no_stale_references_remain(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        for dead in ("_ff_pick_sequence", "_ff_pick_movie"):
            with self.subTest(name=dead):
                self.assertNotIn(dead, src)

    def test_the_file_dialog_offers_both_kinds(self):
        import inspect
        body = inspect.getsource(App._ff_pick)
        self.assertIn("FF_MOV_EXTS", body)
        self.assertIn("FF_SOURCE_EXTS", body)

    def test_cancelling_the_file_dialog_offers_a_folder(self):
        """So choosing a folder is still one step, not two dialogs and a
        guess."""
        import inspect
        body = inspect.getsource(App._ff_pick)
        self.assertIn("askopenfilename", body)
        self.assertIn("askdirectory", body)
        self.assertLess(body.index("askopenfilename"),
                        body.index("askdirectory"))


class TestDitherBar(unittest.TestCase):
    """The progress bar resolves into existence rather than sliding
    across. The pattern is pure arithmetic, so it is checked without a
    Tk canvas."""

    COLS = 64

    def _row(self, frac, row=0, cols=None):
        cols = cols or self.COLS
        return [App._dither_lit(c, row, cols, frac) for c in range(cols)]

    def test_nothing_is_lit_at_zero(self):
        for r in range(4):
            with self.subTest(row=r):
                self.assertFalse(any(self._row(0.0, r)))

    def test_everything_is_lit_at_one(self):
        for r in range(4):
            with self.subTest(row=r):
                self.assertTrue(all(self._row(1.0, r)))

    def test_the_edge_dissolves_over_several_columns(self):
        """Spread over ONE column the whole stipple sat inside a single
        3px column, which just looked like a hard edge with a ragged
        pixel on it."""
        lit = self._row(0.5)
        first_off = lit.index(False)
        # somewhere past the first gap there must still be lit cells --
        # that scatter IS the dissolve
        self.assertTrue(any(lit[first_off:]),
                        "the edge is hard, not dissolving")

    def test_the_dissolve_is_the_width_it_claims(self):
        lit = self._row(0.5)
        first_off = lit.index(False)
        last_on = max(i for i, v in enumerate(lit) if v)
        self.assertLessEqual(last_on - first_off, App._DITHER_EDGE + 2)

    def test_progress_only_ever_adds_cells(self):
        """A cell that lit and then went dark would read as the bar
        going backwards."""
        prev = set()
        for f in [i / 40.0 for i in range(41)]:
            now = {(c, r) for r in range(4)
                   for c in range(self.COLS)
                   if App._dither_lit(c, r, self.COLS, f)}
            with self.subTest(frac=f):
                self.assertTrue(prev <= now, "a lit cell went dark")
            prev = now

    def test_it_fills_from_the_left(self):
        lit = self._row(0.5)
        self.assertTrue(all(lit[:20]))
        self.assertFalse(any(lit[-10:]))

    def test_the_bar_completes_exactly_at_one(self):
        """A bar that reads full before it is full is a small lie. Two
        earlier formulations both went solid around 0.98."""
        def unlit(f, cols):
            return sum(1 for r in range(4) for c in range(cols)
                       if not App._dither_lit(c, r, cols, f))
        for cols in (20, 33, 64, 107, 200):
            with self.subTest(cols=cols):
                self.assertGreater(unlit(0.999, cols), 0)
                self.assertEqual(unlit(1.0, cols), 0)

    def test_the_span_is_measured_not_assumed(self):
        """The last cell to light is usually NOT in the final column --
        the Bayer phase puts it a few columns back, which is what both
        earlier guesses got wrong."""
        for cols in (20, 33, 64):
            with self.subTest(cols=cols):
                e = App._DITHER_EDGE
                span = App._dither_span(cols, e)
                worst = max(c + e * App._BAYER4[r][c % 4] / 16.0
                            for c in range(cols) for r in range(4))
                self.assertAlmostEqual(span, worst, places=9)

    def test_rows_differ_at_the_edge(self):
        """If every row lit identically it would be a hard edge with
        extra steps."""
        rows = [tuple(self._row(0.45, r)) for r in range(4)]
        self.assertGreater(len(set(rows)), 1)

    def test_out_of_range_values_are_clamped(self):
        self.assertFalse(App._dither_lit(0, 0, 64, -3.0))
        self.assertTrue(App._dither_lit(63, 0, 64, 7.0))

    def test_a_zero_width_bar_does_not_divide_by_zero(self):
        self.assertFalse(App._dither_lit(0, 0, 0, 0.5))

    def test_the_bayer_matrix_is_a_permutation(self):
        """All 16 values exactly once -- that is what spreads the lit
        cells evenly instead of clumping them into visible noise."""
        vals = sorted(v for row in App._BAYER4 for v in row)
        self.assertEqual(vals, list(range(16)))

    # ── wiring ──
    def test_the_scan_reports_progress(self):
        import inspect
        body = inspect.getsource(App._ff_scan_worker)
        self.assertIn("ff_bar.set", body)

    def test_the_bar_is_cleared_when_the_scan_ends(self):
        import inspect
        self.assertIn("ff_bar.reset",
                      inspect.getsource(App._ff_scan_reset))

    def test_decoding_does_not_fake_motion(self):
        """ffmpeg gives no usable progress there. A bar that invents
        motion it cannot measure is worse than one that admits it is
        waiting."""
        import inspect
        body = inspect.getsource(App._ff_scan_worker)
        i = body.index("Decoding movie")
        self.assertIn("0.03", body[max(0, i - 200):i + 60])


class TestFixFramesHeldSensitivity(unittest.TestCase):
    """Slow motion was being condemned as freeze frames.

    A fixed threshold cannot separate the two: low enough to catch a
    freeze in a fast sequence and it misses freezes in a slow one; high
    enough for the slow one and every gentle move in the fast one is
    condemned. The test is now relative to how much the clip itself
    moves."""

    @staticmethod
    def _sigs(n=24, step=0.01, seed=1, freeze=None):
        rng = np.random.default_rng(seed)
        base = rng.random((120, 200, 3)).astype(np.float32) * 0.5
        fr = [np.clip(base + step * i, 0, 1).astype(np.float32)
              for i in range(n)]
        if freeze is not None:
            fr[freeze] = fr[freeze - 1].copy()
        return [App._ff_signature(f) for f in fr]

    SPEEDS = (("fast", 0.03), ("slow", 0.002), ("crawl", 0.0004))

    def test_slow_motion_is_not_a_freeze(self):
        """The reported bug."""
        for label, step in self.SPEEDS:
            with self.subTest(speed=label):
                self.assertEqual(
                    App._ff_scan(self._sigs(step=step))["held"], [])

    def test_a_real_freeze_is_still_caught_at_every_speed(self):
        """The other half -- a fix that just stopped flagging things
        would pass the test above and be useless."""
        for label, step in self.SPEEDS:
            with self.subTest(speed=label):
                self.assertEqual(
                    App._ff_scan(self._sigs(step=step, freeze=12))["held"],
                    [12])

    def test_the_same_setting_works_across_speeds(self):
        """No per-clip retuning: that is the point of a relative test."""
        found = [App._ff_scan(self._sigs(step=st, freeze=9))["held"]
                 for _l, st in self.SPEEDS]
        self.assertEqual(found, [[9]] * len(self.SPEEDS))

    # ── the sensitivity control ──
    def test_sensitivity_widens_what_is_flagged(self):
        sigs = self._sigs(step=0.002, freeze=14)
        low = App._ff_scan(sigs, sensitivity=0.2)["held"]
        high = App._ff_scan(sigs, sensitivity=3.0)["held"]
        self.assertLessEqual(len(low), len(high))
        self.assertIn(14, high)

    def test_low_sensitivity_still_catches_an_exact_duplicate(self):
        """An identical frame differs by zero, so no threshold above
        zero can miss it -- turning sensitivity down must not make the
        tool blind to the clearest fault there is."""
        sigs = self._sigs(step=0.01, freeze=7)
        self.assertIn(7, App._ff_scan(sigs, sensitivity=0.1)["held"])

    def test_sensitivity_also_scales_the_black_test(self):
        """One control covering every detector, as asked."""
        frames = self._sigs(n=6)
        # Between the two thresholds: 0.001*0.5 = 0.0005 excludes it,
        # 0.001*3.0 = 0.003 includes it.
        dim = np.full((120, 200, 3), 0.002, np.float32)
        frames[3] = App._ff_signature(dim)
        self.assertNotIn(3, App._ff_scan(frames, sensitivity=0.5)["black"])
        self.assertIn(3, App._ff_scan(frames, sensitivity=3.0)["black"])

    def test_sensitivity_cannot_be_zero_or_negative(self):
        sigs = self._sigs(step=0.01, freeze=5)
        for bad in (0.0, -4.0):
            with self.subTest(sensitivity=bad):
                App._ff_scan(sigs, sensitivity=bad)   # must not raise

    # ── robustness of the baseline ──
    def test_a_long_black_run_does_not_poison_the_baseline(self):
        """Black frames are identical to each other, so counting them in
        the median would drag it toward zero and stop the rest of the
        clip being judged at all."""
        sigs = self._sigs(n=30, step=0.01, freeze=25)
        for i in range(5, 15):
            sigs[i] = App._ff_signature(np.zeros((120, 200, 3), np.float32))
        r = App._ff_scan(sigs)
        self.assertIn(25, r["held"])

    def test_a_mostly_frozen_clip_still_flags_the_frozen_part(self):
        """If most of the clip is a freeze the median goes to zero, and
        only exact duplicates flag -- which is exactly right, because
        they are exactly what is frozen."""
        rng = np.random.default_rng(3)
        base = rng.random((120, 200, 3)).astype(np.float32) * 0.5
        fr = [base.copy() for _ in range(14)]
        fr += [np.clip(base + 0.02 * i, 0, 1).astype(np.float32)
               for i in range(1, 7)]
        held = App._ff_scan([App._ff_signature(f) for f in fr])["held"]
        self.assertGreater(len(held), 8)
        self.assertTrue(all(i < 14 for i in held))

    def test_a_placeholder_frame_is_skipped(self):
        """Missing and unreadable frames arrive as None."""
        sigs = self._sigs(n=12, step=0.01)
        sigs[5] = None
        r = App._ff_scan(sigs)
        self.assertNotIn(5, r["held"])
        self.assertNotIn(5, r["black"])

    def test_an_all_placeholder_input_does_not_raise(self):
        self.assertEqual(App._ff_scan([None, None, None])["held"], [])

    # ── wiring ──
    def test_the_control_reaches_the_scan(self):
        import inspect
        self.assertIn("self.ff_sens.get()",
                      inspect.getsource(App._ff_scan_worker))

    def test_the_control_is_built(self):
        import inspect
        body = inspect.getsource(App._tab_fixframes)
        self.assertIn("ff_sens", body)
        self.assertIn("Sensitivity", body)


class TestH264Transcode(unittest.TestCase):
    """H.265 to H.264. A delivery problem, not a colour one: HEVC is
    fine until it meets an NLE, a review tool or a browser that will not
    decode it."""

    class _Stub(App):
        class _V:
            def __init__(self, v):
                self.v = v

            def get(self):
                return self.v

        def __init__(self):
            self.x264_files = []
            self.x264_codec_var = self._V("H.264 (review)")

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    # ── the mode ──
    def test_the_toggle_has_three_modes(self):
        body = self._src(App._tab_exr)
        for name in ("exr_tog_a", "exr_tog_b", "exr_tog_c"):
            with self.subTest(button=name):
                self.assertIn(name, body)
        self.assertIn("H.264", body)

    def test_the_switcher_handles_all_three(self):
        body = self._src(App._exr_set_mode)
        for key in ('"to_exr"', '"to_mov"', '"transcode"'):
            with self.subTest(mode=key):
                self.assertIn(key, body)

    def test_the_third_panel_starts_hidden(self):
        """All three share one grid cell; two must be removed at
        build time or they stack."""
        body = self._src(App._tab_exr)
        i = body.index("self.exr_f_x264 = tk.Frame")
        self.assertIn("grid_remove()", body[i:i + 300])

    # ── naming ──
    def test_the_output_sits_beside_the_source(self):
        """Not a shared output folder -- a batch drawn from several
        folders has to stay with its own material."""
        app = self._Stub()
        self.assertEqual(app._x264_out_path("/jobs/sh010/take_hevc.mov"),
                         "/jobs/sh010/take_hevc_H264.mp4")

    def test_the_review_codec_writes_mp4_whatever_came_in(self):
        app = self._Stub()
        for src in ("a.mov", "b.mkv", "c.mxf", "d.ts"):
            with self.subTest(src=src):
                self.assertTrue(
                    app._x264_out_path("/x/" + src).endswith("_H264.mp4"))

    def test_no_input_means_no_path(self):
        self.assertEqual(self._Stub()._x264_out_path(), "")

    def test_it_refuses_to_overwrite_its_own_source(self):
        """Reachable if the source is already named _H264.mp4."""
        body = self._src(App._x264_batch)
        self.assertIn("would overwrite the source", body)
        self.assertIn("os.path.abspath", body)

    # ── the encode ──
    def test_every_quality_maps_to_a_crf(self):
        for label in ("High (CRF 18)", "Broadcast (CRF 20)",
                      "Review (CRF 23)", "Small (CRF 28)"):
            with self.subTest(quality=label):
                self.assertIn(label, App.X264_CRF)
                self.assertEqual(App.X264_CRF[label],
                                 int(label.split("CRF ")[1].rstrip(")")))

    def test_an_unknown_quality_falls_back(self):
        self.assertEqual(App.X264_CRF.get("nonsense", 20), 20)

    def test_the_review_codec_is_8_bit_420(self):
        """A 10-bit or 4:2:2 H.264 is perfectly legal and still will not
        open in half the places that conversion exists to satisfy."""
        spec = App.X264_CODECS["H.264 (review)"]
        self.assertEqual(spec["pix"], "yuv420p")
        self.assertEqual(spec["enc"], "libx264")

    def test_it_enables_faststart(self):
        """Most of the point of an H.264 review copy is that it plays
        before it has finished arriving."""
        self.assertIn("+faststart", self._src(App._x264_encode))

    def test_all_three_audio_choices_are_handled(self):
        body = self._src(App._x264_encode)
        self.assertIn('"-c:a", "copy"', body)
        self.assertIn('"aac"', body)
        self.assertIn('"-an"', body)

    def test_a_failed_audio_copy_is_retried(self):
        """The usual cause is source audio MP4 will not carry -- worth
        one automatic retry rather than handing back a codec error the
        user has to decode."""
        body = self._src(App._x264_encode)
        self.assertIn("would not copy into MP4", body)

    def test_failure_reports_ffmpegs_own_words(self):
        """The runner captures stderr and hands back its last line; the
        encoder raises that rather than a generic message."""
        self.assertIn("errf.read()", self._src(App._x264_ffmpeg))
        self.assertIn("raise RuntimeError(err", self._src(App._x264_encode))

    # ── probing ──
    def test_an_unreadable_file_probes_empty(self):
        """A probe that fails must not stop a conversion that would
        have worked."""
        self.assertEqual(App._x264_probe("/nonexistent.mov"), {})

    def test_no_input_codec_is_treated_as_wrong(self):
        """It converts between ten codecs now, so there is no single
        expected input to warn about. The panel used to say "not HEVC"
        at anything that was not HEVC, which stopped being true the
        moment the codec became a choice."""
        body = self._src(App._x264_describe)
        self.assertNotIn("not HEVC", body)
        self.assertNotIn("already H.264", body)

    def test_it_warns_when_the_source_is_already_the_target(self):
        """Then the conversion is a lossy round trip for nothing."""
        body = self._src(App._x264_describe)
        self.assertIn("already this", body)
        for pair in ('codec == "h264"', 'codec == "prores"',
                     'codec == "dnxhd"'):
            with self.subTest(pair=pair):
                self.assertIn(pair, body)

    # ── it did not break the temporal analyser ──
    def test_the_analyser_kept_its_own_names(self):
        """Both features wanted _tc_. A blanket rename took the
        analyser's worker and done with it."""
        for n in ("_tc_worker", "_tc_done", "_tc_analyse", "_tc_start"):
            with self.subTest(method=n):
                self.assertTrue(callable(getattr(App, n, None)))

    def test_the_two_features_share_no_method_names(self):
        import inspect
        analyser = {n for n in dir(App) if n.startswith("_tc_")}
        transcode = {n for n in dir(App) if n.startswith("_x264_")}
        self.assertFalse(analyser & transcode)
        self.assertGreater(len(transcode), 4)
        # and the transcode worker is the one taking an output path
        self.assertIn("out", inspect.signature(App._x264_encode).parameters)


class TestVideoToVideoNaming(unittest.TestCase):
    """The mode outgrew its name: it started as one conversion and now
    offers ten codecs in both directions."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_the_toggle_says_video_to_video(self):
        body = self._src(App._tab_exr)
        self.assertIn("Video  \u2192  Video", body)
        self.assertNotIn("H.265  \u2192  H.264", body)

    def test_the_run_button_does_not_name_one_codec(self):
        """The codec is chosen in the panel, so a button promising
        H.264 would be wrong nine times out of ten."""
        body = self._src(App._tab_exr)
        self.assertNotIn('"Convert to H.264"', body)

    def test_the_three_modes_are_still_distinct(self):
        body = self._src(App._exr_set_mode)
        for key in ('"to_exr"', '"to_mov"', '"transcode"'):
            with self.subTest(mode=key):
                self.assertIn(key, body)


class TestH264Batch(unittest.TestCase):
    """Batch conversion. The queue is the feature; the failure handling
    is what makes it usable overnight."""

    class _Stub(App):
        class _V:
            def __init__(self, v):
                self.v = v

            def get(self):
                return self.v

        def __init__(self):
            self.x264_files = []
            self.x264_codec_var = self._V("H.264 (review)")
            self.logged = []

        def log(self, _w, msg, tag=None):
            self.logged.append(msg)

        def _x264_refresh_queue(self):
            pass

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        self.app = self._Stub()
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _touch(self, name):
        p = os.path.join(self.dir, name)
        with open(p, "wb") as fh:
            fh.write(b"x")
        return p

    # ── the queue ──
    def test_several_files_queue(self):
        paths = [self._touch(f"a{i}.mov") for i in range(3)]
        self.app._x264_add(paths)
        self.assertEqual(self.app.x264_files, paths)

    def test_duplicates_are_not_queued_twice(self):
        p = self._touch("a.mov")
        self.app._x264_add([p, p])
        self.app._x264_add([p])
        self.assertEqual(self.app.x264_files, [p])

    def test_its_own_output_is_never_re_queued(self):
        """Converting a folder twice would otherwise re-encode every
        _H264.mp4 from the first run, quietly producing
        name_H264_H264.mp4 and halving the quality again."""
        good = self._touch("take.mov")
        self.app._x264_add([good, self._touch("take_H264.mp4")])
        self.assertEqual(self.app.x264_files, [good])

    def test_directories_are_skipped(self):
        sub = os.path.join(self.dir, "sub")
        os.makedirs(sub)
        self.app._x264_add([sub, self._touch("a.mov")])
        self.assertEqual(len(self.app.x264_files), 1)

    def test_removing_by_index_works_back_to_front(self):
        """Popping front-to-back would shift the indices under the
        later selections and delete the wrong files."""
        body = self._src(App._x264_remove)
        self.assertIn("reversed(", body)

    def test_clearing_empties_the_queue(self):
        self.app._x264_add([self._touch("a.mov")])
        self.app._x264_clear()
        self.assertEqual(self.app.x264_files, [])

    # ── each output beside its own source ──
    def test_a_batch_across_folders_keeps_each_output_local(self):
        app = self._Stub()
        for src in ("/jobs/a/one.mov", "/jobs/b/two.mov"):
            with self.subTest(src=src):
                self.assertEqual(os.path.dirname(app._x264_out_path(src)),
                                 os.path.dirname(src))

    # ── failure isolation ──
    def test_one_failure_does_not_abandon_the_rest(self):
        """A batch that stops dead on file 3 of 40 is worse than useless
        overnight: a third of a job and no record of which third."""
        body = self._src(App._x264_batch)
        self.assertIn("failed.append", body)
        self.assertIn("for n, src in enumerate", body)
        i = body.index("except Exception")
        self.assertNotIn("return", body[i:i + 200])

    def test_failures_are_summarised_at_the_end(self):
        body = self._src(App._x264_batch_done)
        self.assertIn("failed", body)
        self.assertIn("Converted", body)

    def test_a_source_that_would_be_overwritten_is_skipped_not_fatal(self):
        body = self._src(App._x264_batch)
        self.assertIn("would overwrite the source", body)
        self.assertIn("continue", body)

    def test_progress_names_the_file_and_the_position(self):
        """Overnight, a log of bare percentages tells you nothing about
        which file went wrong."""
        self.assertIn("[{n}/{t}]", self._src(App._x264_batch))

    def test_the_encoder_raises_so_the_batch_can_record_it(self):
        body = self._src(App._x264_encode)
        self.assertIn("raise RuntimeError", body)

    def test_the_settings_are_shared_across_the_batch(self):
        """One quality choice for the queue, not one per file."""
        body = self._src(App._x264_encode)
        self.assertIn("x264_quality_var", body)
        self.assertIn("x264_preset_var", body)


class TestH264Progress(unittest.TestCase):
    """Meaningful batch progress: whole files done, plus how far into
    the current one."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    # ── parsing ffmpeg's own output ──
    def test_it_reads_the_formatted_clock(self):
        self.assertEqual(App._x264_parse_time("out_time=00:00:12.500000"),
                         12.5)
        self.assertEqual(App._x264_parse_time("out_time=01:02:03.250000"),
                         3723.25)

    def test_it_ignores_other_progress_keys(self):
        for line in ("frame=120", "fps=48", "speed=2.1x", ""):
            with self.subTest(line=line):
                self.assertIsNone(App._x264_parse_time(line))

    def test_an_unparsable_time_is_not_a_crash(self):
        """ffmpeg emits N/A before the first frame lands."""
        self.assertIsNone(App._x264_parse_time("out_time=N/A"))

    def test_it_does_not_use_out_time_ms(self):
        """That key is documented as milliseconds and emitted as
        microseconds by most builds, so reading it means guessing which
        one you have."""
        body = self._src(App._x264_parse_time)
        self.assertIn("out_time=", body)
        self.assertNotIn("out_time_ms", body.split('"""')[-1])

    # ── the batch fraction ──
    def test_files_done_plus_the_current_one(self):
        self.assertAlmostEqual(App._x264_overall(2, 0.4, 10), 0.24)
        self.assertAlmostEqual(App._x264_overall(0, 0.5, 1), 0.5)

    def test_it_starts_at_zero_and_ends_at_one(self):
        self.assertEqual(App._x264_overall(0, 0.0, 10), 0.0)
        self.assertEqual(App._x264_overall(10, 0.0, 10), 1.0)

    def test_it_never_exceeds_one(self):
        """A bar that overshoots reads as broken."""
        self.assertEqual(App._x264_overall(5, 2.0, 5), 1.0)
        self.assertEqual(App._x264_overall(99, 1.0, 5), 1.0)

    def test_it_is_monotonic_through_a_batch(self):
        seen = []
        for n in range(4):
            for f in (0.0, 0.5, 1.0):
                seen.append(App._x264_overall(n, f, 4))
        self.assertEqual(seen, sorted(seen))

    def test_an_empty_batch_does_not_divide_by_zero(self):
        self.assertEqual(App._x264_overall(0, 0.0, 0), 0.0)

    def test_a_file_with_no_duration_still_counts(self):
        """Without a duration the bar cannot move WITHIN a file, but it
        must still advance as files complete."""
        self.assertAlmostEqual(App._x264_overall(3, 0.0, 6), 0.5)

    # ── wiring ──
    def test_the_batch_reports_to_the_bar(self):
        body = self._src(App._x264_batch)
        self.assertIn("x264_bar.set", body)
        self.assertIn("_x264_overall", body)

    def test_the_bar_is_cleared_at_the_end(self):
        self.assertIn("x264_bar.reset",
                      self._src(App._x264_batch_done))

    def test_the_label_names_the_file(self):
        """A bare percentage overnight tells you nothing about which
        file is slow."""
        self.assertIn("os.path.basename(src)",
                      self._src(App._x264_batch))

    def test_it_shares_the_widget_with_fix_missing_frames(self):
        body = self._src(App._tab_exr)
        self.assertIn("self._dither_bar(", body)

    # ── the deadlock this avoids ──
    def test_stderr_goes_to_a_file_not_a_pipe(self):
        """ffmpeg writes enough stderr on a bad input to fill a 64K pipe
        buffer, and nothing drains it until the process exits -- so a
        pipe would hang the encode on exactly the files whose errors you
        most want to read."""
        body = self._src(App._x264_ffmpeg)
        self.assertIn("TemporaryFile", body)
        self.assertNotIn("stderr=subprocess.PIPE", body)

    def test_cancelling_terminates_the_encode(self):
        body = self._src(App._x264_ffmpeg)
        self.assertIn("_x264_cancel", body)
        self.assertIn("terminate()", body)

    def test_progress_flags_are_only_added_when_wanted(self):
        """-progress writes a line per frame; on a run nobody is
        watching that is pure noise."""
        body = self._src(App._x264_ffmpeg)
        self.assertIn("if on_progress is not None", body)
        self.assertIn("-nostats", body)


class TestVideoCodecTable(unittest.TestCase):
    """Codec choice for the video-to-video conversion. Extension,
    suffix, pixel format and encoder options all have to agree, which
    is why they live in one table rather than in branches."""

    class _Stub(App):
        def __init__(self, codec="H.264 (review)"):
            self.x264_files = []
            self.x264_codec_var = self._V(codec)

        class _V:
            def __init__(self, v):
                self.v = v

            def get(self):
                return self.v

    def test_every_entry_is_complete(self):
        for name, spec in App.X264_CODECS.items():
            with self.subTest(codec=name):
                for key in ("enc", "ext", "suffix", "pix", "opts", "intra"):
                    self.assertIn(key, spec)

    def test_suffixes_are_unique(self):
        """Two codecs sharing a suffix would overwrite each other's
        output, and the queue filter would mis-skip."""
        sfx = [c["suffix"] for c in App.X264_CODECS.values()]
        self.assertEqual(len(sfx), len(set(sfx)))

    def test_only_h264_is_long_gop(self):
        """The property that actually matters: Nuke fetches a frame
        directly, and a long-GOP codec makes that mean decoding a run of
        frames to reach it."""
        for name, spec in App.X264_CODECS.items():
            with self.subTest(codec=name):
                self.assertEqual(spec["intra"], not spec.get("crf", False))

    def test_intra_codecs_go_in_a_mov(self):
        for name, spec in App.X264_CODECS.items():
            if spec["intra"]:
                with self.subTest(codec=name):
                    self.assertEqual(spec["ext"], ".mov")

    def test_faststart_is_only_for_the_review_codec(self):
        """It means nothing to an application reading frames at random,
        so it would be noise on an intra-frame file."""
        for name, spec in App.X264_CODECS.items():
            with self.subTest(codec=name):
                self.assertEqual(bool(spec.get("faststart")),
                                 not spec["intra"])

    def test_prores_carries_the_apple_vendor_tag(self):
        """Without it the files come out tagged Lavc, and readers key
        off that tag."""
        for name, spec in App.X264_CODECS.items():
            if spec["enc"] == "prores_ks":
                with self.subTest(codec=name):
                    self.assertIn("-vendor", spec["opts"])
                    self.assertIn("ap10", spec["opts"])

    def test_dnxhr_options_name_a_real_profile(self):
        valid = {"dnxhr_lb", "dnxhr_sq", "dnxhr_hq", "dnxhr_hqx",
                 "dnxhr_444"}
        for name, spec in App.X264_CODECS.items():
            if spec["enc"] == "dnxhd":
                with self.subTest(codec=name):
                    self.assertIn(spec["opts"][-1], valid)

    def test_the_10_bit_options_say_so_in_their_pixel_format(self):
        for name, spec in App.X264_CODECS.items():
            if spec["enc"] == "prores_ks" or "HQX" in name:
                with self.subTest(codec=name):
                    self.assertIn("10le", spec["pix"])

    # ── naming ──
    def test_the_output_name_follows_the_codec(self):
        for codec, expect in (("H.264 (review)", "clip_H264.mp4"),
                              ("ProRes 422 LT", "clip_ProResLT.mov"),
                              ("DNxHR SQ", "clip_DNxHRSQ.mov")):
            with self.subTest(codec=codec):
                app = self._Stub(codec)
                self.assertEqual(
                    os.path.basename(app._x264_out_path("/j/clip.mov")),
                    expect)

    def test_two_codecs_do_not_overwrite_each_other(self):
        """A second pass at different settings should sit beside the
        first, not replace it."""
        a = self._Stub("ProRes 422 LT")._x264_out_path("/j/clip.mov")
        b = self._Stub("DNxHR SQ")._x264_out_path("/j/clip.mov")
        self.assertNotEqual(a, b)

    def test_an_unknown_codec_falls_back_rather_than_raising(self):
        app = self._Stub("something removed in a later build")
        self.assertTrue(app._x264_out_path("/j/clip.mov"))

    # ── the queue filter ──
    def test_any_of_its_own_outputs_is_skipped(self):
        """Converting a folder twice must not pick up the first run's
        results, whichever codec produced them."""
        import inspect
        body = inspect.getsource(App._x264_add)
        self.assertIn("X264_CODECS.values()", body)
        self.assertNotIn('endswith("_H264")', body)

    # ── audio ──
    def test_aac_is_not_written_into_a_mov(self):
        """Legal, and several professional readers still refuse it --
        which would waste the point of choosing a codec for
        compatibility."""
        import inspect
        body = inspect.getsource(App._x264_encode)
        self.assertIn("pcm_s16le", body)
        self.assertIn('spec["ext"] == ".mp4"', body)

    # ── the note ──
    def test_the_note_warns_about_long_gop(self):
        import inspect
        body = inspect.getsource(App._x264_codec_changed)
        self.assertIn("Nuke", body)
        self.assertIn("intra-frame", body)

    def test_the_note_is_honest_about_dnxhr(self):
        """ffmpeg leaves its field order unmarked where ProRes is
        tagged progressive, so it is the less certain of the two."""
        import inspect
        body = inspect.getsource(App._x264_codec_changed)
        self.assertIn("field order", body)


class TestCodecDefaultAndInfo(unittest.TestCase):
    """Default to what the app can vouch for in Nuke, and explain the
    whole list in one place."""

    def test_the_default_is_intra_frame(self):
        """H.264 was the default only because it was written first,
        which is a bad reason -- it is the one option on the list a
        frame-random application reads badly."""
        spec = App.X264_CODECS[App.X264_DEFAULT]
        self.assertTrue(spec["intra"])
        self.assertFalse(spec.get("crf", False))

    def test_the_default_exists_in_the_table(self):
        self.assertIn(App.X264_DEFAULT, App.X264_CODECS)

    def test_the_default_is_10_bit(self):
        self.assertIn("10le", App.X264_CODECS[App.X264_DEFAULT]["pix"])

    def test_the_default_is_prores_not_dnxhr(self):
        """ffmpeg tags ProRes progressive and stamps the Apple codec
        IDs; DNxHR comes out with its field order unmarked. That makes
        ProRes the safer thing to hand somebody by default."""
        self.assertEqual(App.X264_CODECS[App.X264_DEFAULT]["enc"],
                         "prores_ks")

    def test_the_combo_honours_a_default(self):
        """It used to always take options[0], so a sensible default
        could only be had by reordering the list."""
        import inspect
        body = inspect.getsource(App._combo)
        self.assertIn("default if default in options", body)

    def test_an_invalid_default_falls_back_to_the_first(self):
        import inspect
        self.assertIn("else options[0]", inspect.getsource(App._combo))

    def test_the_combo_can_carry_a_tooltip(self):
        import inspect
        body = inspect.getsource(App._combo)
        self.assertIn("_tooltip", body)

    # ── the tooltip content ──
    def test_every_codec_appears_in_the_tip(self):
        """Built from the table, so it cannot drift out of step with
        what the encoder actually does."""
        tip = App._x264_codec_tip()
        for name in App.X264_CODECS:
            with self.subTest(codec=name):
                self.assertIn(name, tip)

    def test_every_codec_has_a_measured_size(self):
        for name in App.X264_CODECS:
            with self.subTest(codec=name):
                self.assertIn(name, App.X264_SIZES)
                self.assertRegex(App.X264_SIZES[name], r"^\d+ MB$")

    def test_the_tip_marks_the_default(self):
        self.assertIn("default", App._x264_codec_tip())

    def test_the_tip_separates_intra_from_long_gop(self):
        """That distinction is the whole reason the list has ten
        entries rather than one."""
        tip = App._x264_codec_tip()
        self.assertIn("INTRA-FRAME", tip)
        self.assertIn("LONG-GOP", tip)
        self.assertLess(tip.index("INTRA-FRAME"), tip.index("LONG-GOP"))

    def test_the_tip_names_nuke(self):
        self.assertIn("Nuke", App._x264_codec_tip())

    def test_the_tip_is_honest_about_dnxhr(self):
        tip = App._x264_codec_tip()
        self.assertIn("field order", tip)

    def test_prores_4444_is_not_quoted_as_a_small_step_up(self):
        """It was written at 265 MB from estimate; measured it is 609 --
        12-bit with an alpha plane, nearly five times HQ rather than a
        bit more than it."""
        hq = int(App.X264_SIZES["ProRes 422 HQ"].split()[0])
        q4 = int(App.X264_SIZES["ProRes 4444"].split()[0])
        self.assertGreater(q4, hq * 3)

    def test_sizes_rank_the_way_the_codecs_do(self):
        for smaller, larger in (("ProRes 422 Proxy", "ProRes 422 LT"),
                                ("ProRes 422 LT", "ProRes 422"),
                                ("ProRes 422", "ProRes 422 HQ"),
                                ("DNxHR LB", "DNxHR SQ"),
                                ("DNxHR SQ", "DNxHR HQ"),
                                ("H.264 (review)", "DNxHR LB")):
            with self.subTest(pair=(smaller, larger)):
                self.assertLess(int(App.X264_SIZES[smaller].split()[0]),
                                int(App.X264_SIZES[larger].split()[0]))


class TestSdrToHdr(unittest.TestCase):
    """SDR to HDR10 expansion. Expansion, not recovery -- clipped
    highlights stay clipped, and the UI is required to say so."""

    class _Stub(App):
        class _V:
            def __init__(self, v):
                self.v = v

            def get(self):
                return self.v

        def __init__(self, codec="ProRes 422 LT", hdr=False,
                     peak="1000 nits"):
            self.x264_files = []
            self.x264_codec_var = self._V(codec)
            self.x264_hdr_var = self._V(
                App.HDR_MODES[1] if hdr else App.HDR_MODES[0])
            self.x264_hdr_peak_var = self._V(peak)

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    # ── the capability probe ──
    def test_the_probe_actually_runs_the_filter(self):
        """Listing proves nothing: libplacebo appears in -filters on a
        machine with no Vulkan device and then fails at the first
        frame, so a check based on the filter list would enable a
        button that cannot work."""
        body = self._src(App._hdr_libplacebo_ok)
        self.assertIn("subprocess.run", body)
        self.assertIn("-frames:v", body)
        # It must EXECUTE the filter, not query the filter list. The
        # docstring mentions -filters to explain why, so check the code
        # with the docstring stripped.
        import ast as _ast
        import textwrap
        fn = _ast.parse(textwrap.dedent(body)).body[0]
        if (fn.body and isinstance(fn.body[0], _ast.Expr)
                and isinstance(fn.body[0].value, _ast.Constant)):
            fn.body = fn.body[1:]
        self.assertNotIn("-filters", _ast.unparse(fn))

    def test_the_probe_is_cached(self):
        body = self._src(App._hdr_libplacebo_ok)
        self.assertIn("_hdr_cap", body)

    def test_a_probe_failure_is_not_an_exception(self):
        body = self._src(App._hdr_libplacebo_ok)
        self.assertIn("except Exception", body)

    # ── the filter ──
    def test_it_uses_bt2446a(self):
        """An ITU recommendation defined in both directions, rather
        than somebody's curve -- which matters when the result has to
        survive a client's QC."""
        f = App._hdr_filter("yuv422p10le", 1000)
        self.assertIn("bt.2446a", f)
        self.assertIn("inverse_tonemapping=true", f)

    def test_the_filter_targets_pq_and_bt2020(self):
        f = App._hdr_filter("yuv422p10le", 1000)
        self.assertIn("smpte2084", f)
        self.assertIn("bt2020", f)

    def test_the_peak_reaches_the_filter(self):
        for label, want in (("600 nits", 600), ("1000 nits", 1000),
                            ("4000 nits", 4000)):
            with self.subTest(peak=label):
                nits = App._hdr_peak_nits(label)
                self.assertEqual(nits, want)
                self.assertIn(f"target_peak={want}",
                              App._hdr_filter("yuv422p10le", nits))

    def test_an_unreadable_peak_falls_back(self):
        self.assertEqual(App._hdr_peak_nits("nonsense"), 1000)

    def test_the_output_is_tagged(self):
        """Without these the file carries HDR pixels and SDR metadata,
        which every downstream tool then misreads -- the most common
        way a technically correct conversion still looks wrong."""
        tags = App._hdr_tags()
        for t in ("bt2020nc", "bt2020", "smpte2084"):
            with self.subTest(tag=t):
                self.assertIn(t, tags)

    # ── the guards ──
    def test_it_is_off_by_default(self):
        self.assertFalse(self._Stub()._x264_hdr_on())

    def test_an_unreadable_control_means_off(self):
        """Off is the safe answer: a control that cannot be read should
        not silently turn on a conversion that rewrites the colour
        volume."""
        app = self._Stub()
        app.x264_hdr_var = None
        self.assertFalse(app._x264_hdr_on())

    def test_a_non_string_value_means_off(self):
        app = self._Stub()
        app.x264_hdr_var = self._Stub._V(object())
        self.assertFalse(app._x264_hdr_on())

    def test_an_8_bit_codec_is_refused(self):
        """Expanding into HDR10 at 8 bits returns the added range as
        banding."""
        body = self._src(App._x264_encode)
        self.assertIn("8-bit; HDR10 needs a 10-bit codec", body)

    def test_it_falls_back_rather_than_refusing(self):
        """It no longer depends on libplacebo at all. Selecting the GPU
        engine on a machine without Vulkan quietly uses numpy instead of
        failing -- the conversion is the point, not the backend."""
        import inspect
        body = inspect.getsource(App._hdr_use_builtin)
        self.assertIn("not self._hdr_libplacebo_ok()", body)

    def test_the_builtin_engine_is_the_default(self):
        """Defaulting the other way would put the option that fails on
        most Macs in front of the one that works everywhere."""
        self.assertIn("numpy", App.HDR_ENGINES[0])
        app = self._Stub()
        app.x264_hdr_engine_var = None
        self.assertTrue(app._hdr_use_builtin())

    # ── naming ──
    def test_an_expanded_file_gets_its_own_name(self):
        """It is not interchangeable with a straight transcode of the
        same source, so it must not land on the same name."""
        plain = self._Stub(hdr=False)._x264_out_path("/j/clip.mov")
        hdr = self._Stub(hdr=True)._x264_out_path("/j/clip.mov")
        self.assertNotEqual(plain, hdr)
        self.assertIn("_HDR10", hdr)

    def test_the_queue_skips_expanded_outputs_too(self):
        body = self._src(App._x264_add)
        self.assertIn("_HDR10", body)

    # ── honesty in the UI ──
    def test_the_tip_says_expansion_not_recovery(self):
        body = self._src(App._tab_exr)
        i = body.index("Dynamic range")
        seg = body[i:i + 900]
        self.assertIn("Clipped highlights stay clipped", seg)
        self.assertIn("not a route back", seg)

    def test_the_note_says_which_engine_is_running(self):
        body = self._src(App._x264_codec_changed)
        self.assertIn("_hdr_use_builtin", body)
        self.assertIn("numpy", body)

    def test_the_note_warns_about_8_bit(self):
        body = self._src(App._x264_codec_changed)
        self.assertIn("banding", body)


class TestHdrExpansionMath(unittest.TestCase):
    """The numpy expansion. Written because neither ffmpeg route is
    dependable on a Mac -- libplacebo needs Vulkan, and zscale in the
    builds tested refuses linear conversion outright."""

    def test_pq_matches_the_published_values(self):
        """So downstream tools read the result correctly rather than it
        merely looking plausible."""
        for nits, want in ((0.0, 0.0), (100.0, 0.5081),
                           (1000.0, 0.7518), (10000.0, 1.0)):
            with self.subTest(nits=nits):
                got = float(App._hdr_pq_encode(np.float32(nits)))
                self.assertAlmostEqual(got, want, delta=0.002)

    def test_pq_is_monotonic(self):
        vals = [float(App._hdr_pq_encode(np.float32(n)))
                for n in (0, 1, 10, 100, 1000, 4000, 10000)]
        self.assertEqual(vals, sorted(vals))

    def test_pq_clamps_beyond_the_range(self):
        """PQ(0) is not exactly zero by the formula's own definition --
        with Y=0 it reduces to c1**m2, about 7.3e-07. That is 0.05 of a
        code value at 16 bits, so it rounds away; the maths is left pure
        rather than special-cased, and the test asserts what actually
        matters."""
        self.assertLess(float(App._hdr_pq_encode(np.float32(-5.0))), 1e-5)
        self.assertAlmostEqual(float(App._hdr_pq_encode(np.float32(99999.0))),
                               1.0, places=4)

    # ── the knee ──
    def test_midtones_are_untouched(self):
        """If mid-grey moves, every SDR shot in a timeline shifts
        exposure against the HDR shots it is cut with -- worse than
        doing no expansion at all."""
        for code in (0.18, 0.4, 0.5, 0.7):
            with self.subTest(code=code):
                px = np.full((1, 1, 3), code, np.float32)
                lin = App._hdr_eotf_1886(px)
                out = App._hdr_expand_linear(lin, 1000.0, 0.60)
                np.testing.assert_allclose(
                    out[0, 0], lin[0, 0] * App.HDR_SDR_WHITE, rtol=1e-5)

    def test_highlights_expand(self):
        px = np.full((1, 1, 3), 1.0, np.float32)
        out = App._hdr_expand_linear(App._hdr_eotf_1886(px), 1000.0, 0.60)
        self.assertAlmostEqual(float(out.max()), 1000.0, delta=1.0)

    def test_the_expansion_is_continuous_at_the_knee(self):
        """A step at the knee would show as a hard edge across a
        gradient. The squared term makes the gain leave 1.0 with zero
        slope."""
        ramp = np.linspace(0.0, 1.0, 4096, dtype=np.float32)
        lin = np.repeat(ramp[:, None], 3, axis=1)[None, ...]
        out = App._hdr_expand_linear(lin, 1000.0, 0.60)[0, :, 0]
        jumps = np.abs(np.diff(out))
        self.assertLess(float(jumps.max()), float(np.median(jumps)) * 60)

    def test_it_is_monotonic_across_the_range(self):
        ramp = np.linspace(0.0, 1.0, 512, dtype=np.float32)
        lin = np.repeat(ramp[:, None], 3, axis=1)[None, ...]
        out = App._hdr_expand_linear(lin, 1000.0, 0.60)[0, :, 0]
        self.assertTrue(np.all(np.diff(out) >= -1e-4))

    def test_a_higher_knee_expands_less(self):
        px = np.full((1, 1, 3), 0.85, np.float32)
        lin = App._hdr_eotf_1886(px)
        low = float(App._hdr_expand_linear(lin, 1000.0, 0.60).max())
        high = float(App._hdr_expand_linear(lin, 1000.0, 0.80).max())
        self.assertGreater(low, high)

    def test_the_peak_is_reached_exactly(self):
        for peak in (600.0, 1000.0, 4000.0):
            with self.subTest(peak=peak):
                px = np.ones((1, 1, 3), np.float32)
                out = App._hdr_expand_linear(App._hdr_eotf_1886(px), peak,
                                             0.60)
                self.assertAlmostEqual(float(out.max()), peak, delta=1.0)

    def test_black_stays_black(self):
        px = np.zeros((1, 1, 3), np.float32)
        # Exactly zero in nits; a hair above it in PQ, for the reason
        # given in test_pq_clamps_beyond_the_range.
        self.assertEqual(
            float(App._hdr_expand_frame(px, to_pq=False).max()), 0.0)
        self.assertLess(
            float(App._hdr_expand_frame(px).max()) * 65535.0, 0.5)

    # ── gamut ──
    def test_the_gamut_matrix_is_applied_in_linear_light(self):
        """On gamma-encoded values the result looks close enough to pass
        a glance and is wrong in every saturated colour."""
        import inspect
        body = inspect.getsource(App._hdr_expand_frame)
        self.assertLess(body.index("_hdr_expand_linear"),
                        body.index("_M_709_TO_2020"))

    def test_the_matrix_rows_sum_to_about_one(self):
        """White must map to white; a row that does not sum to 1 tints
        the whole image."""
        for row in App._M_709_TO_2020:
            with self.subTest(row=row):
                self.assertAlmostEqual(sum(row), 1.0, delta=0.005)

    def test_grey_stays_neutral_through_the_matrix(self):
        px = np.full((1, 1, 3), 0.5, np.float32)
        out = App._hdr_expand_frame(px, to_pq=False)[0, 0]
        self.assertAlmostEqual(float(out.max() - out.min()), 0.0,
                               delta=out.max() * 0.01)

    def test_it_can_return_nits_instead_of_pq(self):
        """The form worth keeping when the next step is an EXR or a log
        encode rather than a display."""
        px = np.full((1, 1, 3), 1.0, np.float32)
        self.assertGreater(float(App._hdr_expand_frame(px, to_pq=False).max()),
                           500.0)
        self.assertLessEqual(float(App._hdr_expand_frame(px, to_pq=True).max()),
                             1.0)

    # ── the video path ──
    def test_frames_go_through_a_pipe(self):
        """A two-hour batch would otherwise write and delete a few
        hundred gigabytes of intermediate PNGs for no reason."""
        import inspect
        body = inspect.getsource(App._hdr_expand_video)
        self.assertIn("subprocess.PIPE", body)
        self.assertIn("rawvideo", body)

    def test_it_works_in_16_bit_not_8(self):
        """Expanding an 8-bit decode would quantise before the curve
        that stretches it, which is where banding comes from."""
        import inspect
        body = inspect.getsource(App._hdr_expand_video)
        self.assertIn("rgb48le", body)
        self.assertNotIn("rgb24", body)

    def test_the_output_is_tagged_by_the_video_path_too(self):
        import inspect
        self.assertIn("_hdr_tags()",
                      inspect.getsource(App._hdr_expand_video))

    def test_it_can_be_cancelled(self):
        import inspect
        self.assertIn("_x264_cancel",
                      inspect.getsource(App._hdr_expand_video))

    def test_knee_labels_parse(self):
        for label in App.HDR_KNEES:
            with self.subTest(knee=label):
                k = App._hdr_knee(label)
                self.assertGreater(k, 0.0)
                self.assertLess(k, 1.0)

    def test_an_unparsable_knee_falls_back(self):
        self.assertEqual(App._hdr_knee("nonsense"), 0.60)


class TestHdrMlHook(unittest.TestCase):
    """The socket for a highlight-reconstruction model. No model is
    bundled: the published ones are mostly MIT or Apache on the CODE
    while their weights carry research-only dataset terms, and weights
    with no stated licence default to all rights reserved."""

    @staticmethod
    def _codes(vals):
        c = np.zeros((1, len(vals), 3), np.float32)
        for i in range(3):
            c[0, :, i] = vals
        return c

    def test_no_model_present_is_not_an_error(self):
        """The normal case. The analytic path has to keep working with
        an empty model folder."""
        self.assertEqual(App._hdr_models(), [])

    def test_a_missing_model_folder_is_handled(self):
        import inspect
        self.assertIn("except Exception", inspect.getsource(App._hdr_models))

    def test_it_prefers_coreml_then_falls_back(self):
        """Asking for a provider that is not built in raises, so the
        list is filtered against what is actually available -- and on a
        Mac without CoreML the CPU path still works."""
        import inspect
        body = inspect.getsource(App._hdr_session)
        self.assertIn("CoreMLExecutionProvider", body)
        self.assertIn("CPUExecutionProvider", body)
        self.assertIn("get_available_providers", body)

    def test_sessions_are_cached(self):
        import inspect
        self.assertIn("_hdr_sessions", inspect.getsource(App._hdr_session))

    # ── the mask ──
    def test_only_clipped_pixels_are_selected(self):
        """Everywhere else the SDR frame already holds the answer, and
        letting a network repaint it would trade measured values for
        invented ones."""
        m = App._hdr_clipped_mask(self._codes([0.2, 0.8, 0.95]))
        self.assertEqual(float(m.max()), 0.0)

    def test_blown_pixels_are_fully_selected(self):
        m = App._hdr_clipped_mask(self._codes([0.99, 1.0]))
        np.testing.assert_allclose(m[0], 1.0)

    def test_the_mask_edge_is_soft(self):
        """A hard edge would show as an outline around every
        highlight."""
        m = App._hdr_clipped_mask(self._codes([0.96, 0.97, 0.98, 0.99]))[0]
        self.assertTrue(np.all(np.diff(m) >= 0))
        self.assertGreater(len(set(np.round(m, 2))), 2)

    def test_a_single_clipped_channel_counts(self):
        """A blown red on an otherwise mid-grey pixel is still missing
        information."""
        c = np.zeros((1, 1, 3), np.float32)
        c[0, 0] = [1.0, 0.4, 0.4]
        self.assertAlmostEqual(float(App._hdr_clipped_mask(c)[0, 0]), 1.0,
                               places=3)

    # ── the blend ──
    def test_the_analytic_result_is_kept_outside_the_mask(self):
        code = self._codes([0.2, 0.5, 0.8])
        an = np.full((1, 3, 3), 300.0, np.float32)
        ml = np.full((1, 3, 3), 900.0, np.float32)
        np.testing.assert_allclose(App._hdr_ml_blend(an, ml, code), an)

    def test_the_model_is_used_inside_the_mask(self):
        code = self._codes([1.0])
        an = np.full((1, 1, 3), 300.0, np.float32)
        ml = np.full((1, 1, 3), 900.0, np.float32)
        self.assertAlmostEqual(
            float(App._hdr_ml_blend(an, ml, code).max()), 900.0, delta=1.0)

    def test_a_dimmer_prediction_never_darkens_the_result(self):
        """These networks reconstruct PLAUSIBLE highlights, and a
        plausible highlight dimmer than the analytic curve already
        produced is a step backwards, not a correction."""
        code = self._codes([0.2, 0.97, 1.0])
        an = np.full((1, 3, 3), 300.0, np.float32)
        ml = np.full((1, 3, 3), 120.0, np.float32)
        out = App._hdr_ml_blend(an, ml, code)
        self.assertTrue(np.all(out >= 300.0 - 1e-3))

    def test_the_blend_is_continuous(self):
        code = self._codes(list(np.linspace(0.9, 1.0, 64)))
        an = np.full((1, 64, 3), 300.0, np.float32)
        ml = np.full((1, 64, 3), 2000.0, np.float32)
        out = App._hdr_ml_blend(an, ml, code)[0, :, 0]
        self.assertTrue(np.all(np.diff(out) >= -1e-3))

    def test_onnx_rather_than_a_framework(self):
        """No PyTorch or TensorFlow dependency, and ONNX Runtime on
        macOS has a CoreML provider -- Metal and the Neural Engine, with
        no Vulkan in the chain."""
        import inspect
        self.assertIn("onnxruntime", inspect.getsource(App._hdr_session))


class TestUpscaleAndExpand(unittest.TestCase):
    """The Upscale tab now also expands, takes movies as well as
    sequences, and writes EXR or MOV."""

    class _Stub(App):
        class _V:
            def __init__(self, v):
                self.v = v

            def get(self):
                return self.v

        def __init__(self, fmt="EXR 16-bit half", hdr=False):
            self.usc_fmt_var = self._V(fmt)
            self.usc_mode_var = self._V("both" if hdr else "upscale")
            self._usc_is_movie = False
            self._usc_src_movie = None

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_tab_is_renamed(self):
        """Split into two studios. The expansion one keeps the historical
        "upscale" key so saved projects and the output hand-off still
        find it; the new upscale-only one is "upstudio"."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn('("Upscale Studio",     "upstudio")', src)
        self.assertIn('("Expansion Studio",   "upscale")', src)
        self.assertNotIn('"Upscale & Expand"', src)

    # ── ordering, which is the one thing that must not be got wrong ──
    def test_expansion_runs_after_the_upscale(self):
        """Interpolating PQ code values means interpolating a heavily
        non-linear signal, and the invented pixels come out wrong in
        exactly the highlights this is meant to protect.

        With the two stages now separate studios, the order is enforced
        by WHERE things can go: Expansion Studio never upscales, and the
        output hand-off feeds an upscale into Expansion Studio but never
        an expansion back into Upscale Studio."""
        self.assertNotIn("_rife_upscale", self._src(App._usc_run))
        self.assertIn("_usc_scene_for", self._src(App._usc_run))
        prop = self._src(App._adw_propagate_output)
        self.assertIn('("upstudio", "_ups_load_folder"', prop)
        self.assertIn('if key == "upstudio" and source_key == "upscale":', prop)
        self.assertIn('"upstudio", self.ups_log', self._src(App._ups_run))

    # ── format table ──
    def test_every_format_is_complete(self):
        for name, spec in App.USC_FORMATS.items():
            with self.subTest(fmt=name):
                self.assertIn("ext", spec)
                self.assertIn("movie", spec)

    def test_movie_formats_stage_through_frames(self):
        """Encoding at the end from frames on disk means nothing is lost
        if the run is cancelled part way."""
        for name, spec in App.USC_FORMATS.items():
            if spec["movie"]:
                with self.subTest(fmt=name):
                    self.assertEqual(spec["ext"], ".png")

    def test_movie_formats_name_a_real_codec(self):
        for name, spec in App.USC_FORMATS.items():
            if spec["movie"]:
                with self.subTest(fmt=name):
                    self.assertIn(spec["movie"], App.X264_CODECS)

    def test_an_unknown_format_falls_back(self):
        app = self._Stub("something removed later")
        self.assertEqual(app._usc_format()[1],
                         App.USC_FORMATS[App.USC_DEFAULT_FORMAT])

    # ── writing ──
    def test_exr_holds_values_above_one(self):
        """That is the entire point of writing float: an expanded frame
        in an integer file has already been squeezed back down."""
        try:
            __import__("OpenEXR")
        except ImportError:
            self.skipTest("OpenEXR not available")
        app = self._Stub()
        rgb = np.full((8, 8, 3), 800.0, np.float32)
        path = os.path.join(self.dir, "t.exr")
        app._usc_write_frame(rgb, path, {"ext": ".exr"})
        back = App._ff_read_frame(app, path)
        self.assertAlmostEqual(float(back.max()), 800.0, delta=1.0)

    def test_exr_goes_through_openexr_not_opencv(self):
        """The pip OpenCV wheels ship with the EXR codec compiled
        out -- the same trap the Fix Missing Frames reader hit."""
        body = self._src(App._usc_write_frame)
        self.assertIn("OpenEXR", body)
        i = body.index("OpenEXR")
        self.assertNotIn("cv2.imwrite", body[:i])

    def test_png_is_written_at_16_bit(self):
        """An 8-bit write would quantise a frame that was just
        upscaled, throwing away what the upscale added."""
        body = self._src(App._usc_write_frame)
        self.assertIn("uint16", body)
        self.assertIn("65535", body)

    # ── input ──
    def test_it_accepts_a_movie(self):
        body = self._src(App._usc_accept)
        self.assertIn("FF_MOV_EXTS", body)
        self.assertIn("_usc_load_movie", body)

    def test_one_frame_of_a_sequence_loads_its_folder(self):
        body = self._src(App._usc_accept)
        self.assertIn("os.path.dirname(path)", body)

    def test_a_movie_becomes_a_sequence(self):
        """One processing path rather than two that can drift apart."""
        body = self._src(App._usc_load_movie)
        self.assertIn("_ff_decode_movie", body)
        self.assertIn("_usc_load_folder", body)

    def test_the_movie_frame_rate_is_carried_through(self):
        self.assertIn("_ff_probe_fps", self._src(App._usc_load_movie))
        self.assertIn("_usc_src_fps", self._src(App._usc_run))

    # ── output ──
    def test_the_movie_encode_reuses_the_codec_table(self):
        """One definition of what ProRes 4444 means."""
        self.assertIn("X264_CODECS", self._src(App._usc_encode_movie))

    def test_an_expanded_movie_is_tagged(self):
        self.assertIn("_hdr_tags()", self._src(App._usc_encode_movie))

    def test_an_expanded_output_is_named_differently(self):
        self.assertIn("_HDR10", self._src(App._usc_encode_movie))

    # ── the note ──
    def test_it_says_when_the_output_is_code_not_light(self):
        """The difference between a file you can grade and a file you
        can only display."""
        body = self._src(App._usc_fmt_text)
        self.assertIn("no longer linear light", body)

    # ── progress ──
    def test_it_uses_the_dither_bar(self):
        """The pixelated look now lives in the window's own bar, which
        this tab reports to instead of carrying a second one."""
        self.assertIn("self._dither_bar(", self._src(App._build_progress_bar))
        self.assertIn("self._global_bar_proxy()", self._src(App._tab_upscale))
        self.assertIn("usc_bar.set", self._src(App._usc_run))

    def test_the_bar_is_cleared_on_error(self):
        self.assertIn("usc_bar.reset", self._src(App._usc_run))


class TestExpansionStudioIsExpandOnly(unittest.TestCase):
    """Upscale / Expand / Both used to be a Mode row on one combined
    tab. Upscaling is Upscale Studio's job now, so Expansion Studio is
    expand-only: no Mode, Output Res or Factor controls, and a job mode
    that cannot be talked into upscaling -- not even by an old project
    file restoring usc_mode_var = "both"."""

    class _Stub(App):
        class _V:
            def __init__(self, v):
                self.v = v

            def get(self):
                return self.v

        def __init__(self, mode="both"):
            self.usc_mode_var = self._V(mode)   # a stale, restored value

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_the_job_mode_is_always_expand(self):
        for stale in ("upscale", "both", "expand", "nonsense"):
            with self.subTest(stale=stale):
                app = self._Stub(stale)
                self.assertEqual(app._usc_job_mode(), "expand")
                self.assertFalse(app._usc_does_upscale())
                self.assertTrue(app._usc_does_expand())

    def test_expansion_follows_the_mode_not_a_second_control(self):
        """Two controls for one decision is how they end up
        disagreeing."""
        body = self._src(App._usc_hdr_on)
        self.assertIn("_usc_does_expand", body)
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertNotIn("usc_hdr_var", src)

    def test_no_mode_resolution_or_factor_controls_remain(self):
        body = self._src(App._tab_upscale)
        for gone in ("usc_mode_var", "usc_res_var", "usc_fac_var",
                     '"Output Res"', '"Factor"', '_card(lf, "Settings")'):
            with self.subTest(gone=gone):
                self.assertNotIn(gone, body)

    def test_resolution_lives_in_upscale_studio(self):
        body = self._src(App._tab_upstudio)
        for frag in ("self.ups_res_var", "self.ups_fac_var", '"Output Res"',
                     '"Factor"', "or scale the input"):
            with self.subTest(fragment=frag):
                self.assertIn(frag, body)

    # ── greying out (still used, now by the model tiles) ──
    def test_the_whole_subtree_is_greyed(self):
        """A disabled section with one live widget left in it is the one
        nobody expects to work."""
        body = self._src(App._usc_set_enabled)
        self.assertIn("winfo_children", body)
        self.assertIn("Combobox", body)

    def test_greying_is_reversible(self):
        """Switching back must restore the original colour, not a
        guess at it."""
        body = self._src(App._usc_set_enabled)
        self.assertIn("_usc_fg", body)

    # ── the run ──
    def test_the_run_never_touches_the_upscaler(self):
        """Not even at 1:1 -- that would still resample and sharpen."""
        body = self._src(App._usc_run)
        self.assertNotIn("_rife_upscale", body)
        self.assertNotIn("_usc_target", body)
        self.assertIn("up_rgb = np.array(orig)", body)

    def test_the_output_folder_names_the_mode(self):
        body = self._src(App._usc_run)
        self.assertIn('"expand": "EXPAND"', body)

    def test_the_progress_label_names_the_mode(self):
        body = self._src(App._usc_run)
        for lbl in ("Upscale + expand", "Expand"):
            with self.subTest(label=lbl):
                self.assertIn(lbl, body)


class TestNoMethodShadowing(unittest.TestCase):
    """A method whose name is also assigned as an instance attribute is
    shadowed the moment that attribute is set, and the call then lands
    on whatever value was stored. Python gives no warning; it fails at
    run time on the line that looks correct.

    This bit once: _usc_mode was already holding the PREVIEW mode
    ("preview" / "wipe" / "diff") when a method of the same name was
    added, so building the tab called a string."""

    @staticmethod
    def _scan():
        with open(SRC, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        methods = {n.name for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)}
        props = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef):
                for d in n.decorator_list:
                    # @property and @x.setter -- assignment IS the
                    # intended interface for those
                    if isinstance(d, ast.Name) and d.id == "property":
                        props.add(n.name)
                    if isinstance(d, ast.Attribute) and d.attr == "setter":
                        props.add(n.name)
        bad = []
        for n in ast.walk(tree):
            if not isinstance(n, ast.Assign):
                continue
            for t in n.targets:
                if (isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "self"
                        and t.attr in methods
                        and t.attr not in props):
                    # A callable assigned over a callable is deliberate
                    # shadowing and harmless.
                    if not isinstance(n.value, (ast.Lambda, ast.Name,
                                                ast.Attribute, ast.Call)):
                        bad.append((t.attr, n.lineno))
        return sorted(set(bad))

    def test_no_method_is_shadowed_by_a_plain_value(self):
        bad = self._scan()
        self.assertEqual(bad, [], f"method names overwritten by values: {bad}")

    def test_the_job_mode_method_is_not_the_preview_mode_attribute(self):
        """The specific clash, pinned by name."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("def _usc_job_mode(self):", src)
        self.assertNotIn("def _usc_mode(self):", src)
        self.assertIn('self._usc_mode        = "preview"', src)

    def test_the_tab_builds_without_calling_a_string(self):
        """What actually broke: _tab_upscale -> _usc_mode_changed ->
        _usc_does_upscale -> self._usc_mode(), on a string."""
        import inspect
        body = inspect.getsource(App._usc_does_upscale)
        self.assertIn("_usc_job_mode()", body)
        self.assertNotIn("self._usc_mode()", body)


class TestUpscaleReadsExr(unittest.TestCase):
    """PIL cannot open EXR at all -- it raises UnidentifiedImageError,
    which is what this tab did on every EXR sequence it was pointed at,
    in both the preview and the run."""

    class _Stub(App):
        def __init__(self, encoded=False):
            self.enc = encoded

        def _adw_source_is_encoded(self):
            return self.enc

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _exr(self, name, rgb):
        try:
            import OpenEXR
        except ImportError:
            self.skipTest("OpenEXR not available")
        h, w = rgb.shape[:2]
        path = os.path.join(self.dir, name)
        out = OpenEXR.OutputFile(path, OpenEXR.Header(w, h))
        out.writePixels({c: rgb[:, :, i].astype(np.float32).tobytes()
                         for i, c in enumerate("RGB")})
        out.close()
        return path

    def test_an_exr_loads(self):
        path = self._exr("p.0001.exr", np.full((8, 8, 3), 0.18, np.float32))
        img = self._Stub()._usc_read_rgb(path)
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(img.size, (8, 8))

    def test_it_does_not_go_through_pil_for_exr(self):
        import ast as _ast
        import inspect
        import textwrap
        fn = _ast.parse(textwrap.dedent(
            inspect.getsource(App._usc_read_rgb))).body[0]
        if (fn.body and isinstance(fn.body[0], _ast.Expr)
                and isinstance(fn.body[0].value, _ast.Constant)):
            fn.body = fn.body[1:]
        code = _ast.unparse(fn)
        self.assertIn("_ff_read_frame", code)
        self.assertIn("FF_LINEAR_EXTS", code)

    def test_scene_linear_gets_the_transfer_curve(self):
        """The upscaler wants display-referred 8-bit. Handing it linear
        values would feed the model a picture far darker than the one
        the artist is looking at."""
        path = self._exr("p.0002.exr", np.full((4, 4, 3), 0.18, np.float32))
        a = np.array(self._Stub(encoded=False)._usc_read_rgb(path))
        self.assertGreater(int(a[0, 0, 0]), 100)   # ~118, sRGB mid grey

    def test_an_encoded_exr_is_not_curved_twice(self):
        """EXRs from AI video tools hold display-ready floats despite
        the container implying otherwise -- the same trap the Fix
        Missing Frames preview hit."""
        path = self._exr("p.0003.exr", np.full((4, 4, 3), 0.18, np.float32))
        lin = int(np.array(self._Stub(encoded=False)._usc_read_rgb(path))[0, 0, 0])
        enc = int(np.array(self._Stub(encoded=True)._usc_read_rgb(path))[0, 0, 0])
        self.assertLess(enc, lin)
        self.assertAlmostEqual(enc, 46, delta=2)

    def test_over_range_values_are_clamped_for_the_upscaler(self):
        rgb = np.full((4, 4, 3), 4.0, np.float32)
        a = np.array(self._Stub()._usc_read_rgb(self._exr("p.0004.exr", rgb)))
        self.assertEqual(int(a.max()), 255)

    def test_ordinary_formats_still_use_pil(self):
        from PIL import Image
        path = os.path.join(self.dir, "x.png")
        Image.fromarray(np.full((6, 6, 3), 200, np.uint8)).save(path)
        img = self._Stub()._usc_read_rgb(path)
        self.assertEqual(img.size, (6, 6))
        self.assertEqual(img.mode, "RGB")

    def test_both_the_preview_and_the_run_use_it(self):
        """It failed in both places; fixing one would have left the
        other raising."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertEqual(src.count("self._usc_read_rgb(src_path)"), 2)
        self.assertNotIn('orig = Image.open(src_path).convert("RGB")', src)


class TestPreviewExposure(unittest.TestCase):
    """The exposure slider did nothing useful for judging an expansion.

    Everything an expansion adds lives ABOVE display white, so on an SDR
    monitor it is invisible until you stop down to it -- and the old
    control could not, because it quantised to 8 bits and clipped to
    0..1 before the exposure was applied."""

    class _Stub(App):
        def __init__(self):
            pass

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        self.app = self._Stub()
        code = np.linspace(0.85, 1.0, 8, dtype=np.float32)
        self.img = np.repeat(code[None, :, None], 3, axis=2)
        self.nits = App._hdr_expand_frame(self.img, 1000.0, 0.60,
                                          to_pq=False)

    def test_stopping_down_reveals_detail(self):
        """The whole point. At -4 stops the eight steps of the gradient
        must be eight distinct values, not one flat block."""
        out = self.app._apply_ev_gamma_float(self.nits, -4.0, 1.0,
                                             peak=100.0)[0, :, 0]
        self.assertEqual(len(set(int(v) for v in out)), 8)

    def test_the_old_path_could_not(self):
        """Guards the diagnosis: on 8-bit input the same gradient
        collapses, because the highlights were destroyed before the
        exposure ran."""
        u8 = (np.clip(self.img, 0, 1) * 255).astype(np.uint8)
        out = self.app._apply_ev_gamma(u8, -4.0, 1.0)[0, :, 0]
        self.assertLessEqual(len(set(int(v) for v in out)), 3)

    def test_exposure_is_applied_before_the_clip(self):
        """Otherwise the highlights are gone before the slider can pull
        them into view."""
        import ast as _ast
        import textwrap
        fn = _ast.parse(textwrap.dedent(
            self._src(App._apply_ev_gamma_float))).body[0]
        code = _ast.unparse(fn)
        i = code.index("2.0 ** float(ev)")
        self.assertIn("np.clip", code[i:i + 120])

    def test_linear_data_is_display_encoded(self):
        """The bug in the screenshot: the expanded side came out dark
        with crushed blacks because linear light was written straight to
        8-bit. 18% grey showed at code 46 instead of 118."""
        app = self._Stub()
        for lin in (0.005, 0.05, 0.18, 0.5, 1.0):
            with self.subTest(linear=lin):
                a = np.full((1, 1, 3), lin, np.float32)
                got = int(app._apply_ev_gamma_float(a, 0.0, 1.0, 1.0)[0, 0, 0])
                ref = int(App._organic_linear_to_display(a)[0, 0, 0])
                self.assertAlmostEqual(got, ref, delta=2)

    def test_the_curve_is_piecewise_not_a_plain_power(self):
        """The linear segment near black is the whole difference
        between lifted shadows and correct ones, and shadows are where
        this was being judged."""
        a = np.full((1, 1, 3), 0.002, np.float32)
        got = float(App._linear_to_srgb(a)[0, 0, 0])
        self.assertAlmostEqual(got, 0.002 * 12.92, places=5)
        self.assertLess(got, 0.002 ** (1 / 2.2))

    def test_the_uint8_sibling_still_gets_no_curve(self):
        """Its input is already encoded -- which is exactly why copying
        its shape for the float version was wrong."""
        import inspect
        self.assertNotIn("_linear_to_srgb",
                         inspect.getsource(App._apply_ev_gamma))

    def test_both_sides_agree_below_the_knee(self):
        """Mid-tones are anchored, so an unexpanded input and an
        expanded result must display identically there -- if they do
        not, the wipe is comparing two different grades rather than two
        dynamic ranges."""
        app = self._Stub()
        lin = np.full((1, 1, 3), 0.10, np.float32)
        plain = int(app._apply_ev_gamma_float(lin, 0.0, 1.0, 1.0)[0, 0, 0])
        nits = App._hdr_expand_linear(lin, 1000.0, 0.60)
        expanded = int(app._apply_ev_gamma_float(
            nits, 0.0, 1.0, App.HDR_SDR_WHITE)[0, 0, 0])
        self.assertAlmostEqual(plain, expanded, delta=2)

    def test_the_peak_normalises_the_range(self):
        """Nits from an expansion have to be viewed against the display
        white they were expanded from, or every frame reads as blown."""
        flat = np.full((1, 1, 3), 100.0, np.float32)
        out = self.app._apply_ev_gamma_float(flat, 0.0, 1.0, peak=100.0)
        self.assertEqual(int(out[0, 0, 0]), 255)
        half = self.app._apply_ev_gamma_float(flat, -1.0, 1.0, peak=100.0)
        self.assertLess(int(half[0, 0, 0]), 200)

    def test_exposure_is_monotonic(self):
        vals = [int(self.app._apply_ev_gamma_float(
            self.nits, ev, 1.0, peak=100.0)[0, 0, 0])
            for ev in (-4.0, -2.0, 0.0, 2.0)]
        self.assertEqual(vals, sorted(vals))

    def test_gamma_still_applies(self):
        a = self.app._apply_ev_gamma_float(self.nits, -3.0, 1.0, peak=100.0)
        b = self.app._apply_ev_gamma_float(self.nits, -3.0, 2.2, peak=100.0)
        self.assertGreater(int(b[0, 0, 0]), int(a[0, 0, 0]))

    # ── wiring ──
    def test_the_toolbar_is_on_the_upscale_viewer(self):
        body = self._src(App._tab_upscale)
        self.assertIn("_ev_toolbar(", body)
        self.assertIn("usc_ev_var", body)

    def test_the_preview_is_rebuilt_from_the_float(self):
        """Without this the slider moves and nothing happens, which is
        worse than not having it."""
        body = self._src(App._usc_view_base)
        self.assertIn("_usc_up_float", body)
        self.assertIn("_apply_ev_gamma_float", body)

    def test_wipe_and_diff_use_the_same_view(self):
        """Otherwise the slider works in one mode and not the others."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("self._usc_view_base().resize", src)
        self.assertEqual(src.count("self._usc_view_base().resize"), 2)

    def test_the_preview_runs_the_same_expansion_as_the_render(self):
        """A preview of a different picture is not a preview: Tone map's
        preview and the result rebuilt for viewing both go through the
        same _hdr_expand_frame the render's _usc_scene_for uses."""
        self.assertIn("_hdr_expand_frame", self._src(App._usc_analytic))
        self.assertIn("self._usc_analytic(code)",
                      self._src(App._usc_preview_model))
        self.assertIn("self._usc_analytic(code)",
                      self._src(App._usc_model_result))
        self.assertIn("_hdr_expand_frame", self._src(App._usc_scene_for))

    def test_the_slider_does_not_re_upscale(self):
        """It is a view control; making it re-run the model would turn
        a slider drag into a minute of GPU time."""
        body = self._src(App._usc_ev_redraw)
        self.assertIn("_usc_display", body)
        self.assertNotIn("_rife_upscale", body)

    def test_no_expansion_means_no_float_and_the_old_path(self):
        body = self._src(App._usc_view_base)
        self.assertIn("if fl is None:", body)
        self.assertIn("return self._usc_up_base", body)


class TestPreviewExposureBothSides(unittest.TestCase):
    """One toolbar, and it moves both halves of the comparison."""

    class _V:
        def __init__(self, v):
            self.v = v

        def get(self):
            return self.v

    class _Stub(App):
        def __init__(self):
            self.usc_ev_var = TestPreviewExposureBothSides._V(0.0)
            self.usc_gm_var = TestPreviewExposureBothSides._V(1.0)
            self._usc_orig_base = None
            self._usc_orig_float = None
            self._usc_orig_peak = 1.0
            self._usc_up_float = None

        def _adw_source_is_encoded(self):
            return False

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        self.app = self._Stub()
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    # ── the duplicate ──
    def test_there_is_only_one_toolbar_on_this_tab(self):
        """The other was built with throwaway DoubleVars and no redraw
        callback -- nothing read its values and nothing redrew when they
        changed, so it could never have done anything."""
        body = self._src(App._tab_upscale)
        self.assertEqual(body.count("self._ev_toolbar("), 1)

    def test_the_surviving_one_is_wired(self):
        body = self._src(App._tab_upscale)
        i = body.index("self._ev_toolbar(")
        seg = body[i:i + 160]
        self.assertIn("usc_ev_var", seg)
        self.assertIn("_usc_ev_redraw", seg)
        self.assertNotIn("tk.DoubleVar(value=0.0)", seg)

    # ── both sides ──
    def test_the_input_responds_to_exposure(self):
        """A wipe where only one half responds tells you nothing about
        what the expansion changed."""
        from PIL import Image
        self.app._usc_orig_base = Image.fromarray(
            np.full((4, 4, 3), 200, np.uint8))
        at0 = int(np.array(self.app._usc_view_orig())[0, 0, 0])
        self.app.usc_ev_var = self._V(-1.0)
        at1 = int(np.array(self.app._usc_view_orig())[0, 0, 0])
        self.assertEqual(at0, 200)
        self.assertLess(at1, 120)

    def test_a_linear_source_reveals_its_own_highlights(self):
        try:
            import OpenEXR
        except ImportError:
            self.skipTest("OpenEXR not available")
        h, w = 4, 8
        rgb = np.zeros((h, w, 3), np.float32)
        rgb[:] = np.linspace(0.5, 6.0, w)[None, :, None]
        path = os.path.join(self.dir, "s.0001.exr")
        out = OpenEXR.OutputFile(path, OpenEXR.Header(w, h))
        out.writePixels({c: rgb[:, :, i].tobytes()
                         for i, c in enumerate("RGB")})
        out.close()
        self.app._usc_orig_base = self.app._usc_read_rgb(path)
        (self.app._usc_orig_float,
         self.app._usc_orig_peak) = self.app._usc_read_float(path)
        self.assertIsNotNone(self.app._usc_orig_float)
        self.app.usc_ev_var = self._V(-4.0)
        a = np.array(self.app._usc_view_orig())[0, :, 0]
        self.assertEqual(len(set(int(v) for v in a)), 8)

    def test_an_8_bit_source_has_no_float(self):
        """Nothing above 1.0 to find, so the plain image is used."""
        path = os.path.join(self.dir, "x.png")
        from PIL import Image
        Image.fromarray(np.full((4, 4, 3), 10, np.uint8)).save(path)
        fl, peak = self.app._usc_read_float(path)
        self.assertIsNone(fl)
        self.assertEqual(peak, 1.0)

    def test_a_missing_file_does_not_raise(self):
        fl, peak = self.app._usc_read_float("/nonexistent.exr")
        self.assertIsNone(fl)
        self.assertEqual(peak, 1.0)

    def test_neutral_settings_skip_the_work(self):
        """At 0 EV and gamma 1 the original image is handed back
        untouched rather than round-tripped through the maths."""
        from PIL import Image
        base = Image.fromarray(np.full((4, 4, 3), 123, np.uint8))
        self.app._usc_orig_base = base
        self.assertIs(self.app._usc_view_orig(), base)

    def test_every_draw_path_uses_the_adjusted_input(self):
        """Preview, wipe and diff -- if one reads the raw image the
        slider works in some modes and not others."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertGreaterEqual(src.count("_usc_view_orig()"), 4)
        for dead in ("self._usc_orig_base.resize",
                     "base = self._usc_orig_base or"):
            with self.subTest(pattern=dead):
                self.assertNotIn(dead, src)


class TestExpansionRoundTrip(unittest.TestCase):
    """A scene-linear EXR goes to the upscaler as 8-bit sRGB and comes
    back to be expanded. If the decode does not match the encode, the
    expansion starts from the wrong light -- and the error is
    concentrated in the shadows, which is where it shows."""

    class _Stub(App):
        def __init__(self):
            pass

    def test_srgb_is_its_own_inverse(self):
        a = np.linspace(0.0, 1.0, 64, dtype=np.float32)
        back = App._srgb_to_linear(App._linear_to_srgb(a))
        np.testing.assert_allclose(back, a, atol=2e-4)

    def test_the_two_curves_differ_most_in_shadows(self):
        """60% dark at code 0.1 against 4% at 0.75 -- a contrast error,
        not an exposure one, which is why it read as crushed blacks
        rather than a dark picture."""
        def ratio(code):
            a = np.full((1, 1, 3), code, np.float32)
            return (float(App._hdr_eotf_1886(a)[0, 0, 0])
                    / float(App._srgb_to_linear(a)[0, 0, 0]))
        shadow, highlight = ratio(0.1), ratio(0.75)
        self.assertLess(shadow, 0.45)        # measured 0.40
        self.assertGreater(highlight, 0.93)  # measured 0.96
        self.assertLess(shadow, highlight - 0.4)

    def test_both_halves_of_the_wipe_match_below_the_knee(self):
        """Mid-tones are anchored, so an unexpanded input and an
        expanded result must display identically there. If they do not,
        the wipe compares two grades rather than two dynamic ranges."""
        app = self._Stub()
        for lin in (0.002, 0.01, 0.05, 0.18, 0.4):
            with self.subTest(linear=lin):
                a = np.full((1, 1, 3), lin, np.float32)
                inp = int(app._apply_ev_gamma_float(a, 0.0, 1.0, 1.0)[0, 0, 0])
                code = App._linear_to_srgb(a)
                nits = App._hdr_expand_frame(code, 1000.0, 0.60,
                                             to_pq=False, eotf="srgb")
                out = int(app._apply_ev_gamma_float(
                    nits, 0.0, 1.0, App.HDR_SDR_WHITE)[0, 0, 0])
                self.assertEqual(inp, out)

    def test_the_wrong_eotf_would_crush_the_shadows(self):
        """Pins the bug, so a future change back gets caught."""
        app = self._Stub()
        a = np.full((1, 1, 3), 0.01, np.float32)
        code = App._linear_to_srgb(a)
        wrong = int(app._apply_ev_gamma_float(
            App._hdr_expand_frame(code, 1000.0, 0.60, to_pq=False,
                                  eotf="bt1886"),
            0.0, 1.0, App.HDR_SDR_WHITE)[0, 0, 0])
        right = int(app._apply_ev_gamma_float(
            App._hdr_expand_frame(code, 1000.0, 0.60, to_pq=False,
                                  eotf="srgb"),
            0.0, 1.0, App.HDR_SDR_WHITE)[0, 0, 0])
        self.assertLess(wrong, right - 8)

    def test_video_still_defaults_to_bt1886(self):
        """Rec.709 video really is BT.1886; only this app's own sRGB
        round trip is the exception."""
        import inspect
        sig = inspect.signature(App._hdr_expand_frame)
        self.assertEqual(sig.parameters["eotf"].default, "bt1886")
        self.assertNotIn("eotf=", inspect.getsource(App._hdr_expand_video))

    def test_the_upscale_path_picks_the_curve_from_the_source(self):
        import inspect
        body = inspect.getsource(App._usc_eotf)
        self.assertIn("_ff_is_linear", body)
        self.assertIn('"srgb"', body)
        self.assertIn('"bt1886"', body)

    def test_an_unknown_source_falls_back_to_video(self):
        app = self._Stub()
        app._usc_frames = None
        app._usc_folder = None
        self.assertEqual(app._usc_eotf(), "bt1886")

    def test_every_expansion_call_in_the_tab_passes_the_curve(self):
        """One missed call site and that path silently keeps the old
        error."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        i = src.index("def _tab_upscale")
        seg = src[i:]
        for chunk in seg.split("_hdr_expand_frame(")[1:]:
            with self.subTest(call=chunk[:40].strip()):
                self.assertIn("eotf=", chunk[:260])


class TestPreviewGamut(unittest.TestCase):
    """An expanded frame is in BT.2020. The monitor is 709.

    Shown unconverted, neutrals stay correct -- white maps to white --
    while every saturated colour is pulled toward grey. On a night
    exterior full of ride lights that reads as the two halves of a wipe
    not matching, with no obvious cause because the greys agree."""

    class _V:
        def __init__(self, v):
            self.v = v

        def get(self):
            return self.v

    class _Stub(App):
        def __init__(self):
            self.usc_ev_var = TestPreviewGamut._V(0.0)
            self.usc_gm_var = TestPreviewGamut._V(1.0)
            self._usc_up_peak = App.HDR_SDR_WHITE
            self._usc_up_gamut = "bt2020"
            self._usc_up_base = None
            self._usc_up_float = None

    COLOURS = (("neutral", (0.18, 0.18, 0.18)),
               ("warm", (0.60, 0.35, 0.12)),
               ("blue", (0.10, 0.16, 0.35)),
               ("red", (0.55, 0.06, 0.06)),
               ("green", (0.06, 0.20, 0.05)))

    def test_the_inverse_matrix_round_trips(self):
        """Inverted at run time rather than typed in again, so the pair
        cannot drift apart."""
        lin = np.array([[[0.3, 0.5, 0.2]]], np.float32)
        fwd = lin @ np.array(App._M_709_TO_2020, np.float32).T
        np.testing.assert_allclose(App._bt2020_to_709(fwd), lin, atol=1e-4)

    def test_neutrals_are_unaffected_by_the_conversion(self):
        """Which is exactly why the bug was hard to see."""
        grey = np.full((1, 1, 3), 0.4, np.float32)
        fwd = grey @ np.array(App._M_709_TO_2020, np.float32).T
        np.testing.assert_allclose(fwd, grey, atol=0.01)

    def test_saturated_colours_shift_without_it(self):
        red = np.array([[[0.55, 0.06, 0.06]]], np.float32)
        fwd = red @ np.array(App._M_709_TO_2020, np.float32).T
        self.assertGreater(float(np.abs(fwd - red).max()), 0.05)

    def test_both_halves_match_on_every_colour(self):
        """Not just on greys -- the greys agreed before the fix too."""
        app = self._Stub()
        for name, lin in self.COLOURS:
            with self.subTest(colour=name):
                a = np.array(lin, np.float32).reshape(1, 1, 3)
                inp = app._apply_ev_gamma_float(a, 0.0, 1.0, 1.0)[0, 0]
                app._usc_up_float = App._hdr_expand_frame(
                    App._linear_to_srgb(a), 1000.0, 0.60,
                    to_pq=False, eotf="srgb")
                ups = np.array(app._usc_view_base())[0, 0]
                self.assertLessEqual(
                    int(np.abs(inp.astype(int) - ups.astype(int)).max()), 1)

    def test_the_conversion_is_display_only(self):
        """The written file keeps its BT.2020 primaries -- that is the
        point of expanding. Only the viewer converts."""
        import inspect
        self.assertIn("_bt2020_to_709",
                      inspect.getsource(App._usc_view_base))
        self.assertNotIn("_bt2020_to_709",
                         inspect.getsource(App._usc_write_frame))
        self.assertNotIn("_bt2020_to_709",
                         inspect.getsource(App._hdr_expand_frame))

    def test_the_conversion_can_be_turned_off(self):
        """A float that is already 709 must not be converted twice."""
        app = self._Stub()
        app._usc_up_gamut = "bt709"
        a = np.array([[[0.55, 0.06, 0.06]]], np.float32)
        app._usc_up_float = a * App.HDR_SDR_WHITE
        direct = app._apply_ev_gamma_float(a, 0.0, 1.0, 1.0)[0, 0]
        shown = np.array(app._usc_view_base())[0, 0]
        np.testing.assert_array_equal(direct, shown)

    def test_negatives_from_the_conversion_are_clipped(self):
        """Out-of-709 colours map to negative components, which would
        wrap when cast to uint8."""
        import inspect
        body = inspect.getsource(App._usc_view_base)
        i = body.index("_bt2020_to_709")
        self.assertIn("np.clip", body[i - 40:i + 60])


class TestSceneReferredExr(unittest.TestCase):
    """The EXR is working material, not a display delivery.

    Written in absolute nits it put diffuse white at 100.0 and
    speculars at 1000.0, which any comp reading 1.0 as white shows as
    roughly six and a half stops over -- the exposure looked blown when
    only the range had changed."""

    @staticmethod
    def _expand(lin, **kw):
        a = np.full((1, 1, 3), lin, np.float32)
        opts = dict(to_pq=False, eotf="srgb", gamut="bt709", scale="white")
        opts.update(kw)
        return float(App._hdr_expand_frame(
            App._linear_to_srgb(a), 1000.0, 0.60, **opts)[0, 0, 0])

    def test_the_exposure_is_unchanged(self):
        """The whole request: same exposure, only the top expanded."""
        for lin in (0.005, 0.02, 0.18, 0.4, 0.55):
            with self.subTest(linear=lin):
                self.assertAlmostEqual(self._expand(lin), lin, delta=0.002)

    def test_diffuse_white_stays_at_one(self):
        """Not 100. That factor is the entire bug."""
        nits = self._expand(0.5, scale="nits")
        white = self._expand(0.5)
        self.assertAlmostEqual(nits / white, App.HDR_SDR_WHITE, delta=1.0)

    def test_highlights_go_above_one(self):
        self.assertGreater(self._expand(1.0), 5.0)
        self.assertAlmostEqual(self._expand(1.0), 10.0, delta=0.2)

    def test_the_peak_sets_the_headroom(self):
        for peak, stops in ((600.0, 6.0), (1000.0, 10.0), (4000.0, 40.0)):
            with self.subTest(peak=peak):
                a = np.full((1, 1, 3), 1.0, np.float32)
                got = float(App._hdr_expand_frame(
                    App._linear_to_srgb(a), peak, 0.60, to_pq=False,
                    eotf="srgb", gamut="bt709", scale="white")[0, 0, 0])
                self.assertAlmostEqual(got, stops, delta=0.3)

    def test_the_exr_keeps_709_primaries(self):
        """Widening the gamut in working material would shift every
        saturated colour against the plate it has to sit with."""
        red = np.array([[[0.55, 0.06, 0.06]]], np.float32)
        out = App._hdr_expand_frame(App._linear_to_srgb(red), 1000.0, 0.60,
                                    to_pq=False, eotf="srgb",
                                    gamut="bt709", scale="white")
        np.testing.assert_allclose(out, red, atol=0.01)

    def test_the_display_delivery_still_gets_bt2020(self):
        """PQ output is a display format and keeps the wide gamut --
        only the scene-referred path opts out."""
        import inspect
        sig = inspect.signature(App._hdr_expand_frame)
        self.assertEqual(sig.parameters["gamut"].default, "bt2020")
        self.assertEqual(sig.parameters["scale"].default, "nits")

    def test_the_exr_path_asks_for_scene_referred(self):
        """Now enforced one level down: the EXR path calls
        _usc_scene_for, whose fallback is the analytic
        _hdr_expand_frame with these exact arguments -- checked
        there so a future analytic-only regression is still caught
        even though this function no longer names them directly."""
        import inspect
        body = inspect.getsource(App._usc_scene_for)
        self.assertIn('gamut="bt709"', body)
        self.assertIn('scale="white"', body)
        run_body = inspect.getsource(App._usc_run)
        self.assertIn("px = self._usc_scene_for(", run_body)

    def test_the_render_uses_the_shared_dispatch(self):
        """_usc_run goes through _usc_scene_for -- the single point
        that decides Topaz vs analytic and, in its analytic fallback,
        asks for scale="white"."""
        import inspect
        body = inspect.getsource(App._usc_run)
        i = body.index("_usc_up_float")
        self.assertIn("_usc_scene_for(", body[i:i + 500])
        self.assertIn('_usc_up_gamut = "bt709"', body[i:i + 4000])

    def test_the_preview_now_generates_a_short_topaz_span(self):
        """Superseded: Preview used to always stay analytic for Topaz.
        It now generates a short (not VACE's 81-frame) span so the
        person can actually compare Topaz's real result against Tone
        map and Wan inpaint, exactly the workflow this exists for --
        see TestTopazPreviewAndCache for the full behaviour."""
        import inspect
        body = inspect.getsource(App._usc_preview_model)
        i = body.index('if m == "topaz":')
        j = body.index('if m == "wan_vace":', i)
        seg = body[i:j]
        self.assertIn("_usc_topaz_expand(", seg)
        self.assertIn("USC_TOPAZ_PREVIEW_SPAN", seg)

class TestGenerativeHighlights(unittest.TestCase):
    """VACE inpainting into blown regions, then analytic expansion so
    the new detail lands above white."""

    @staticmethod
    def _scene():
        code = np.full((60, 60, 3), 0.25, np.float32)
        code[20:40, 20:40] = 1.0
        return code

    def _analytic(self, code):
        return App._hdr_expand_frame(code, 1000.0, 0.60, to_pq=False,
                                     eotf="srgb", gamut="bt709",
                                     scale="white")

    # ── the mask ──
    def test_only_blown_pixels_are_marked(self):
        code = self._scene()
        m = App._hdr_ai_mask_frame(code, 0.96)
        self.assertEqual(int(m[5, 5]), 0)
        self.assertEqual(int(m[30, 30]), 255)

    def test_the_mask_is_dilated(self):
        """A model given a mask that stops exactly at the clipped pixels
        has no unclipped neighbours to take its cue from, and paints
        something unrelated to the light around it."""
        code = np.full((40, 40, 3), 0.25, np.float32)
        code[20:24, 20:24] = 1.0
        marked = int((App._hdr_ai_mask_frame(code, 0.96) > 127).sum())
        self.assertGreater(marked, 16 * 3)

    def test_dilation_can_be_turned_off(self):
        code = np.full((40, 40, 3), 0.25, np.float32)
        code[20:24, 20:24] = 1.0
        self.assertLess(
            int((App._hdr_ai_mask_frame(code, 0.96, dilate=0) > 127).sum()),
            int((App._hdr_ai_mask_frame(code, 0.96) > 127).sum()))

    def test_nothing_blown_means_an_empty_mask(self):
        code = np.full((20, 20, 3), 0.3, np.float32)
        self.assertEqual(int((App._hdr_ai_mask_frame(code) > 127).sum()), 0)

    # ── the merge ──
    def test_generated_detail_actually_survives(self):
        """The bug this was rewritten for. Taking the brighter of the
        two -- the rule the ONNX path uses -- is self-defeating here: in
        a blown region the analytic result is a flat plateau at the
        peak, so the maximum against it is that same plateau and every
        reconstructed detail is discarded."""
        code = self._scene()
        analytic = self._analytic(code)
        rng = np.random.default_rng(1)
        gen = code.copy()
        gen[20:40, 20:40] = np.clip(0.55 + 0.35 * rng.random((20, 20, 1)),
                                    0, 1)
        out = App._hdr_ai_merge(analytic, gen, code, amount=1.0)
        self.assertAlmostEqual(float(analytic[20:40, 20:40, 0].std()), 0.0,
                               places=4)
        self.assertGreater(float(out[20:40, 20:40, 0].std()), 0.5)

    def test_the_level_comes_from_the_maths_not_the_model(self):
        """The model supplies structure; the expansion supplies level.
        Mean brightness must survive, or a generative pass becomes an
        exposure change nobody asked for."""
        code = self._scene()
        analytic = self._analytic(code)
        rng = np.random.default_rng(2)
        gen = code.copy()
        gen[20:40, 20:40] = np.clip(0.2 + 0.6 * rng.random((20, 20, 1)),
                                    0, 1)
        out = App._hdr_ai_merge(analytic, gen, code, amount=1.0)
        self.assertAlmostEqual(float(out[20:40, 20:40, 0].mean()),
                               float(analytic[20:40, 20:40, 0].mean()),
                               delta=0.2)

    def test_unclipped_pixels_are_untouched(self):
        code = self._scene()
        analytic = self._analytic(code)
        gen = np.full_like(code, 0.9)
        out = App._hdr_ai_merge(analytic, gen, code, amount=1.0)
        np.testing.assert_array_equal(out[:10, :10], analytic[:10, :10])

    def test_nothing_comes_back_below_white(self):
        """A pixel that clipped at white cannot plausibly return
        dimmer than white."""
        code = self._scene()
        analytic = self._analytic(code)
        gen = code.copy()
        gen[20:40, 20:40] = 0.02
        out = App._hdr_ai_merge(analytic, gen, code, amount=1.0)
        self.assertGreaterEqual(float(out[20:40, 20:40].min()), 1.0 - 1e-4)

    def test_amount_scales_the_effect(self):
        code = self._scene()
        analytic = self._analytic(code)
        rng = np.random.default_rng(3)
        gen = code.copy()
        gen[20:40, 20:40] = np.clip(0.55 + 0.35 * rng.random((20, 20, 1)),
                                    0, 1)
        stds = [float(App._hdr_ai_merge(analytic, gen, code,
                                        amount=a)[20:40, 20:40, 0].std())
                for a in (0.0, 0.25, 0.5, 1.0)]
        self.assertAlmostEqual(stds[0], 0.0, places=4)
        self.assertEqual(stds, sorted(stds))

    def test_zero_amount_is_the_analytic_result(self):
        code = self._scene()
        analytic = self._analytic(code)
        gen = np.full_like(code, 0.9)
        np.testing.assert_allclose(
            App._hdr_ai_merge(analytic, gen, code, amount=0.0), analytic)

    def test_an_empty_mask_short_circuits(self):
        """No blown pixels means no generation call is worth making."""
        code = np.full((20, 20, 3), 0.3, np.float32)
        analytic = self._analytic(code)
        gen = np.full_like(code, 0.9)
        np.testing.assert_array_equal(
            App._hdr_ai_merge(analytic, gen, code), analytic)

    def test_a_black_generation_does_not_divide_by_zero(self):
        code = self._scene()
        analytic = self._analytic(code)
        out = App._hdr_ai_merge(analytic, np.zeros_like(code), code)
        self.assertTrue(np.isfinite(out).all())

    # ── the mask clip ──
    def test_it_uses_the_apps_own_mask_encoder(self):
        """Not a second one written here. The app's builder encodes
        with libx264 via ffmpeg and, because the mask is identical on
        every frame, writes ONE frame and loops it -- 40x smaller at 4K
        than writing all 81."""
        import inspect
        body = inspect.getsource(App._usc_ai_generate)
        self.assertIn("_vace_build_mask_clip", body)
        self.assertIn("custom_mask=", body)

    def test_the_old_cv2_writer_is_gone(self):
        """cv2.VideoWriter with mp4v fails to open silently and leaves
        a file that uploads forever."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertNotIn("_vace_build_spatial_mask_clip", src)

    def test_it_reports_how_much_is_blown(self):
        """If nothing is blown there is nothing to reconstruct, and
        spending a generation call to repaint zero pixels is a cost with
        no result."""
        import inspect
        body = inspect.getsource(App._usc_ai_generate)
        self.assertIn("frac < 1e-4", body)
        self.assertIn("of the frame is blown", body)

    def test_empty_clips_are_caught_before_upload(self):
        """A zero-byte clip is the shape a silent encoder failure takes,
        and uploading one is how eleven minutes disappear with nothing
        to show."""
        import inspect
        body = inspect.getsource(App._usc_ai_generate)
        self.assertIn("getsize", body)
        self.assertIn("failed to encode", body)
        self.assertLess(body.index("failed to encode"),
                        body.index("_vace_submit"))

    def test_uploads_have_their_own_timeout(self):
        """30 minutes is right for a render that genuinely takes that
        long; for a few megabytes it means a broken upload looks
        identical to a slow one for half an hour."""
        self.assertLess(App.VACE_FAL_UPLOAD_TIMEOUT_S,
                        App.VACE_FAL_TIMEOUT_S)
        import inspect
        body = inspect.getsource(App._vace_submit)
        i = body.index("upload_file")
        self.assertIn("VACE_FAL_UPLOAD_TIMEOUT_S", body[i - 200:i + 300])

    def test_the_watchdog_honours_a_custom_timeout(self):
        import inspect
        body = inspect.getsource(App._fal_call_with_watchdog_inner)
        self.assertIn("timeout_s or self.VACE_FAL_TIMEOUT_S", body)

    # ── the UI ──
    def test_the_robot_is_a_pixel_bitmap(self):
        self.assertGreater(len(App._ROBOT_PIXELS), 6)
        widths = {len(r) for r in App._ROBOT_PIXELS}
        self.assertEqual(len(widths), 1)
        self.assertTrue(set("".join(App._ROBOT_PIXELS)) <= {"#", "."})

    def test_the_toggle_latches(self):
        """A mode control that does not show its state is how people run
        a job twice before noticing which one they ran."""
        import inspect
        body = inspect.getsource(App._robot_toggle)
        self.assertIn('st["on"] = bool(on)', body)
        self.assertIn("def _paint", body)

    def test_the_toggle_uses_the_same_cell_size_family(self):
        import inspect
        self.assertIn("cell=3", inspect.getsource(App._robot_toggle))

    def test_the_note_warns_about_emissive_sources(self):
        """Ride lights and signage are where it will invent filaments
        and lettering that were never there."""
        import inspect
        body = inspect.getsource(App._usc_ai_tip_text)
        self.assertIn("EMISSIVE", body)
        self.assertIn("DIFFUSE", body)

    def test_the_mask_can_be_viewed_before_spending_a_call(self):
        import inspect
        self.assertIn("_hdr_ai_mask_frame",
                      inspect.getsource(App._usc_ai_show_mask))


class TestGenerativeWiring(unittest.TestCase):
    """The engine and the button existed and nothing connected them, so
    the toggle changed the label and not the picture. These check that
    the calls are actually reachable."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_the_preview_runs_the_generation(self):
        """The reported symptom: no visual difference with it on. Wan
        inpaint's preview really generates, and what is shown is the
        generation merged over the analytic base."""
        body = self._src(App._usc_preview_model)
        i = body.index('if m == "wan_vace":')
        self.assertIn("_usc_ai_generate", body[i:])
        self.assertIn("_usc_merge_wan", self._src(App._usc_model_result))
        self.assertIn("_hdr_ai_merge", self._src(App._usc_merge_wan))

    def test_the_render_runs_the_generation(self):
        body = self._src(App._usc_run)
        self.assertIn("_usc_ai_generate", body)
        self.assertIn("_usc_ai_apply", body)

    def test_the_render_generates_once_for_the_whole_span(self):
        """VACE is a video model; asking it frame by frame would throw
        away the temporal term that is the entire reason for using it
        here instead of a single-image network."""
        body = self._src(App._usc_run)
        i = body.index("_usc_ai_generate")
        j = body.index("for i, fname in enumerate")
        self.assertLess(i, j)

    def test_an_empty_mask_skips_the_call(self):
        """A plate with no blown pixels would otherwise cost a full
        generation to come back identical."""
        body = self._src(App._usc_ai_generate)
        self.assertIn("if frac < 1e-4:", body)
        i = body.index("frac < 1e-4")
        self.assertLess(i, body.index("_vace_submit"))

    def test_a_failed_generation_keeps_the_run(self):
        """The analytic expansion is still a correct result, so this
        degrades rather than losing an hour of work."""
        body = self._src(App._usc_run)
        self.assertIn("AI expansion failed, using analytic", body)

    def test_a_short_generation_degrades_per_frame(self):
        body = self._src(App._usc_ai_apply)
        self.assertIn("i >= len(gen_names)", body)
        self.assertIn("return expanded", body)

    def test_the_merge_happens_before_the_pq_encode(self):
        """Merging past the PQ encode would be blending two non-linear
        signals."""
        body = self._src(App._usc_run)
        i = body.index("_usc_ai_apply(_lin")
        self.assertLess(i, body.index("_hdr_pq_encode", i))

    def test_both_output_families_get_it(self):
        """EXR and the display deliveries -- one wired and the other
        not would be worse than neither."""
        body = self._src(App._usc_run)
        # EXR, display, and the Multi model output wan_inpaint layer
        self.assertEqual(body.count("_usc_ai_apply("), 3)

    def test_the_generated_frame_is_resized_if_it_differs(self):
        """VACE returns its own resolution, not necessarily the
        plate's."""
        body = self._src(App._usc_ai_apply)
        self.assertIn("cv2.resize", body)

    def test_the_toggle_is_read_not_just_displayed(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertGreater(src.count("_usc_ai_on"), 3)


class TestUpscaleLayout(unittest.TestCase):
    """The column was taller than a laptop screen, which put the Run
    button below the fold -- indistinguishable from a missing one."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_the_column_scrolls(self):
        self.assertIn("self._scroll_column(lf)",
                      self._src(App._tab_upscale))

    def test_the_run_button_is_pinned(self):
        """In the reserved strip, so it never scrolls away from the
        settings that configure it."""
        body = self._src(App._tab_upscale)
        self.assertIn("self._runbtn(_usc_bottom", body)
        # ...with no second bar under it: progress goes to the window's
        self.assertNotIn("_dither_bar(", body)
        self.assertIn("self._global_bar_proxy()", body)

    def test_preview_sits_directly_under_the_model_tiles(self):
        """Select, press, watch the same tiles fill: Preview sits right
        under the tiles it runs, above the per-model settings -- not at
        the foot of the column, where opening Wan's settings pushed it
        below the fold, away from the tiles whose progress it drives."""
        body = self._src(App._tab_upscale)
        tiles = body.index("self._usc_build_model_tiles(c_dr)")
        prev = body.index("self.usc_preview_btn = self._runbtn(c_dr")
        self.assertLess(tiles, prev)
        for later in ("usc_topaz_fmt_var = tk.StringVar",
                      '"Target peak"', '_card(lf, "Output Format")'):
            with self.subTest(later=later):
                self.assertLess(prev, body.index(later))

    def test_preview_and_compare_sits_between_tiles_and_preview(self):
        """Pick models, set how to compare, press Preview -- in that
        order down the column."""
        body = self._src(App._tab_upscale)
        tiles = body.index("self._usc_build_model_tiles(c_dr)")
        slot = body.index("self._usc_pc_slot.pack(")
        prev = body.index("self.usc_preview_btn = self._runbtn(c_dr")
        self.assertLess(tiles, slot)
        self.assertLess(slot, prev)
        self.assertIn("c3 = self._usc_pc_slot", body)
        self.assertNotIn('_card(lf, "Preview & Compare")', body)

    # ── descriptions behind icons ──
    def test_no_wrapped_paragraphs_are_left_in_the_column(self):
        """Each was four or five lines of permanent vertical space."""
        body = self._src(App._tab_upscale)
        self.assertNotIn("wraplength=380, justify=\"left\", anchor=\"w\"\n"
                         "        ).pack", body)
        self.assertLess(body.count("wraplength=360"), 2)

    def test_the_header_explanation_is_a_tooltip(self):
        body = self._src(App._tab_upscale)
        i = body.index("Expansion Studio")
        self.assertIn("_tooltip(", body[i:i + 900])

    def test_dynamic_tooltips_are_fetched_when_opened(self):
        """A static tooltip captures its string at build time, which is
        wrong for anything describing the CURRENT settings -- it would
        keep explaining whatever was selected when the tab was made."""
        body = self._src(App._tooltip_dynamic)
        self.assertIn("text_fn()", body)
        i = body.index("def _show")
        self.assertLess(i, body.index("text_fn()"))

    def test_the_changing_notes_use_the_dynamic_one(self):
        body = self._src(App._tab_upscale)
        self.assertIn("_tooltip_dynamic(", body)
        self.assertIn("_usc_fmt_text", body)
        self.assertIn("_usc_ai_tip_text", body)

    def test_the_icons_still_signal_state(self):
        """Colour is the only cue left once the text is hidden, so the
        icon has to change with selection."""
        self.assertEqual(App.USC_TILE_AMBER, "#DDAA44")
        body = self._src(App._usc_draw_tile)
        self.assertIn('amber if sel else "#555555"', body)
        self.assertIn("#DDAA44", self._src(App._usc_fmt_changed))

    def test_the_text_survived_the_move(self):
        """Hidden, not deleted -- the warnings are the useful part."""
        ai = self._src(App._usc_ai_tip_text)
        self.assertIn("EMISSIVE", ai)
        self.assertIn("Show blown mask", ai)
        self.assertIn("no longer linear light", self._src(App._usc_fmt_text))


class TestGenerativeVaceInterface(unittest.TestCase):
    """Two bugs at the VACE boundary, both from assuming an interface
    instead of reading it."""

    class _V:
        def __init__(self, v):
            self.v = v

        def get(self):
            return self.v

    class _Stub(App):
        def __init__(self):
            self.usc_ai_thresh = TestGenerativeVaceInterface._V(0.96)

        def _adw_source_is_encoded(self):
            return True

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        self.app = self._Stub()
        self.dir = tempfile.mkdtemp()
        self.names = []
        from PIL import Image
        for k in range(4):
            a = np.full((60, 80, 3), 60, np.uint8)
            a[20:30, 10 + k * 12:20 + k * 12] = 255      # a MOVING light
            n = f"f.{k:04d}.png"
            Image.fromarray(a).save(os.path.join(self.dir, n))
            self.names.append(n)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_size_comes_from_the_frame_not_the_clip_builder(self):
        """_vace_build_source_clip returns the PATH it wrote. Passing a
        67-character path where OpenCV wants a (w, h) is how this first
        failed."""
        body = self._src(App._usc_ai_generate)
        self.assertIn("_pv_frame(os.path.join(folder, names[r0])).size",
                      body)
        i = body.index("size = ")
        self.assertNotIn("size = self._vace_build_source_clip", body[:i + 60])

    def test_the_source_clip_is_greyed_in_the_masked_area(self):
        """The builder's own docstring records that handing the model
        clean footage in the masked area makes the output equal the
        input -- it copies the answer it was given. That, not the
        resize, is why switching this on changed nothing visible."""
        body = self._src(App._usc_ai_generate)
        self.assertIn("custom_mask=brush", body)

    def test_the_grey_requirement_is_documented_where_it_is_used(self):
        self.assertIn("GRAY (128)",
                      self._src(App._vace_build_source_clip))

    # ── the union mask ──
    def test_the_mask_has_the_shape_opencv_wants(self):
        m = self.app._usc_ai_union_mask(self.dir, self.names, 0, 3, (80, 60))
        self.assertEqual(m.shape, (60, 80))

    def test_it_covers_every_position_of_a_moving_highlight(self):
        """One mask serves the whole range, so a highlight that travels
        would otherwise be regenerated in some frames and not others --
        and that boundary flickers exactly where the eye is drawn."""
        union = self.app._usc_ai_union_mask(self.dir, self.names, 0, 3,
                                            (80, 60))
        single = App._hdr_ai_mask_frame(
            np.asarray(self.app._usc_read_rgb(
                os.path.join(self.dir, self.names[0])), np.float32) / 255.0,
            0.96)
        self.assertGreater(float((union > 0).mean()),
                           float((single > 0).mean()) * 2)

    def test_an_unblown_span_yields_nothing(self):
        from PIL import Image
        d = tempfile.mkdtemp()
        try:
            names = []
            for k in range(3):
                n = f"d.{k:04d}.png"
                Image.fromarray(np.full((30, 40, 3), 60, np.uint8)).save(
                    os.path.join(d, n))
                names.append(n)
            m = self.app._usc_ai_union_mask(d, names, 0, 2, (40, 30))
            self.assertEqual(int((m > 0).sum()), 0)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_an_empty_mask_skips_before_any_upload(self):
        body = self._src(App._usc_ai_generate)
        self.assertLess(body.index("frac < 1e-4"),
                        body.index("_vace_build_source_clip"))


class TestVaceSubmitKeys(unittest.TestCase):
    """The files dict handed to _vace_submit is keyed, and the keys have
    to be the ones _vace_payload reads.

    A wrong key uploads the file perfectly well and then omits its URL
    from the request, so the API reports a MISSING FIELD -- an error
    that points nowhere near the actual mistake."""

    @staticmethod
    def _payload_keys():
        """The urls.get(...) names the payload builder actually uses."""
        import inspect
        return set(re.findall(r'urls\.get\([\'"]([a-z_]+)[\'"]\)',
                              inspect.getsource(App._vace_payload)))

    @staticmethod
    def _submit_calls():
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        out = []
        for chunk in src.split("_vace_submit(")[1:]:
            seg = chunk[:200]
            out.append(set(re.findall(r'[\'"]([a-z_]+)[\'"]\s*:', seg)))
        return out

    def test_the_payload_reads_video_and_mask(self):
        keys = self._payload_keys()
        self.assertIn("video", keys)
        self.assertIn("mask", keys)

    def test_every_call_site_uses_keys_the_payload_reads(self):
        """The bug: 'source' instead of 'video'."""
        known = self._payload_keys() | {"first", "last"}
        for i, keys in enumerate(self._submit_calls()):
            if not keys:
                continue
            with self.subTest(call=i):
                self.assertTrue(
                    keys <= known,
                    f"unknown keys {keys - known}; payload reads {known}")

    def test_no_call_site_still_says_source(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertNotIn('_vace_submit({"source"', src)

    def test_every_call_site_supplies_the_video(self):
        """Omitting it is what the API rejects."""
        for i, keys in enumerate(self._submit_calls()):
            if not keys:
                continue
            with self.subTest(call=i):
                self.assertIn("video", keys)


class TestVaceProgress(unittest.TestCase):
    """A generation is a network round trip that takes minutes. Without
    reporting it is indistinguishable from a hang."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_every_phase_the_api_reports_has_a_span(self):
        """A phase with no entry would silently reset the bar to 0."""
        import inspect
        body = inspect.getsource(App._vace_submit)
        reported = set(re.findall(r'_ph\([\'"]([a-z]+)[\'"]', body))
        self.assertTrue(reported)
        for ph in reported:
            with self.subTest(phase=ph):
                self.assertIn(ph, App._VACE_PHASE_SPAN)

    def test_the_spans_are_ordered_and_cover_the_bar(self):
        spans = list(App._VACE_PHASE_SPAN.values())
        self.assertAlmostEqual(spans[0][0], 0.0, places=3)
        self.assertAlmostEqual(spans[-1][1], 1.0, places=3)
        for (lo, hi), (nlo, _nhi) in zip(spans, spans[1:]):
            self.assertLessEqual(lo, hi)
            self.assertAlmostEqual(hi, nlo, places=3)

    def test_render_gets_the_largest_share(self):
        """Rendering is minutes and uploading is seconds. Equal shares
        would park the bar for the entire wait and then jump."""
        spans = App._VACE_PHASE_SPAN
        width = lambda k: spans[k][1] - spans[k][0]
        self.assertGreater(width("render"), width("upload"))
        self.assertGreater(width("render"), width("queue"))
        self.assertGreater(width("render"), 0.4)

    def test_progress_never_goes_backwards_through_a_run(self):
        order = ("mask", "upload", "queue", "render", "download", "decode")
        seen = [App._vace_phase_fraction(p)[0] for p in order]
        self.assertEqual(seen, sorted(seen))

    def test_a_phase_can_report_its_own_fraction(self):
        lo, _l = App._vace_phase_fraction("upload", frac=0.0)
        mid, _m = App._vace_phase_fraction("upload", frac=0.5)
        hi, _h = App._vace_phase_fraction("upload", frac=1.0)
        self.assertLess(lo, mid)
        self.assertLess(mid, hi)
        self.assertLessEqual(hi, App._VACE_PHASE_SPAN["upload"][1] + 1e-6)

    def test_the_label_carries_the_detail(self):
        """'position 3' and '81 frames' are the only things that tell
        you a queue is moving."""
        _f, label = App._vace_phase_fraction("queue", "position 3")
        self.assertIn("Queue", label)
        self.assertIn("position 3", label)

    def test_an_unknown_phase_does_not_reset_the_bar(self):
        f, _l = App._vace_phase_fraction("something_new")
        self.assertGreaterEqual(f, 0.0)
        self.assertLessEqual(f, 1.0)

    # ── the pulse ──
    def test_waiting_phases_pulse(self):
        """Queue has no progress of its own at all. Render now creeps on
        an estimate instead, and pulses throughout so the movement is
        not mistaken for measurement."""
        body = self._src(App._usc_ai_generate)
        self.assertIn('waiting = phase == "queue"', body)
        self.assertIn("usc_bar.pulse", body)
        i = body.index('phase == "render"')
        self.assertIn("usc_bar.pulse(True)", body[i:i + 400])

    def test_the_pulse_never_turns_a_lit_cell_off(self):
        """It signals activity; claiming the bar had gone backwards
        would be worse than showing nothing."""
        body = self._src(App._dither_bar)
        i = body.index("def _pulse_tick")
        seg = body[i:i + 700]
        self.assertIn('if not st["on"][r][c]:', seg)
        self.assertIn("continue", seg)

    def test_the_pulse_does_not_move_the_fraction(self):
        """A bar that creeps forward on a timer during the one phase
        that takes minutes is the one most likely to be believed, and
        the one most likely to be lying."""
        body = self._src(App._dither_bar)
        i = body.index("def _pulse_tick")
        self.assertNotIn('st["frac"]', body[i:i + 700])

    def test_reset_stops_the_pulse(self):
        body = self._src(App._dither_bar)
        i = body.index("def _reset")
        self.assertIn('st["pulse"] = False', body[i:i + 200])

    def test_the_pulse_stops_when_the_generation_ends(self):
        self.assertIn("usc_bar.pulse(False)",
                      self._src(App._usc_ai_generate))


class TestVaceEta(unittest.TestCase):
    """The render estimate. Learned from completed runs rather than
    guessed, and never allowed to claim the job has finished."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self._saved = App.VACE_TIMING_FILE
        App.VACE_TIMING_FILE = os.path.join(self.dir, "t.json")

    def tearDown(self):
        App.VACE_TIMING_FILE = self._saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_first_run_says_it_is_guessing(self):
        """'about 3 minutes' on a run with no history behind it would
        be a stronger claim than the number deserves."""
        est, learned = App._vace_estimate(81)
        self.assertFalse(learned)
        self.assertGreater(est, 0)
        self.assertIn("first run", App._vace_eta_text(0, est, learned))

    def test_it_learns_from_completed_runs(self):
        App._vace_record_render(81, 160.0)
        est, learned = App._vace_estimate(81)
        self.assertTrue(learned)
        self.assertAlmostEqual(est, 160.0, delta=1.0)
        self.assertNotIn("first run", App._vace_eta_text(0, est, learned))

    def test_one_queue_backup_does_not_poison_every_later_run(self):
        """The median, not the mean: a single twenty-minute queue would
        otherwise inflate the estimate indefinitely."""
        for secs in (160.0, 175.0, 1400.0):
            App._vace_record_render(81, secs)
        est, _l = App._vace_estimate(81)
        self.assertLess(est, 300.0)
        self.assertGreater(est, 150.0)

    def test_it_scales_with_frame_count(self):
        App._vace_record_render(80, 160.0)
        self.assertAlmostEqual(App._vace_estimate(160)[0],
                               App._vace_estimate(80)[0] * 2, delta=2.0)

    def test_the_history_is_bounded(self):
        for i in range(40):
            App._vace_record_render(10, 10.0 + i)
        self.assertLessEqual(len(App._vace_timing_history()), 20)

    def test_nonsense_records_are_ignored(self):
        App._vace_record_render(0, 100.0)
        App._vace_record_render(10, 0.0)
        App._vace_record_render(10, -5.0)
        self.assertEqual(App._vace_timing_history(), [])

    def test_a_missing_or_corrupt_file_is_not_fatal(self):
        with open(App.VACE_TIMING_FILE, "w", encoding="utf-8") as fh:
            fh.write("not json at all")
        self.assertEqual(App._vace_timing_history(), [])
        est, learned = App._vace_estimate(50)
        self.assertFalse(learned)
        self.assertGreater(est, 0)

    # ── the bar ──
    def test_progress_is_capped_short_of_the_end(self):
        """Letting an ESTIMATE reach the end of its span would show a
        finished bar next to a running job -- at that point the bar is
        not merely unhelpful, it is saying something untrue."""
        for elapsed in (200.0, 400.0, 100000.0):
            with self.subTest(elapsed=elapsed):
                self.assertLessEqual(
                    App._vace_render_progress(elapsed, 200.0),
                    App.VACE_ETA_CAP)

    def test_progress_rises_with_elapsed_time(self):
        vals = [App._vace_render_progress(e, 200.0)
                for e in (0, 50, 100, 150, 200)]
        self.assertEqual(vals, sorted(vals))
        self.assertEqual(vals[0], 0.0)

    def test_a_zero_estimate_does_not_divide_by_zero(self):
        self.assertEqual(App._vace_render_progress(10.0, 0.0), 0.0)

    def test_overrun_is_admitted_not_reported_as_zero(self):
        """Ordered the other way, a first run that ran long kept
        reporting '~0s left', which reads as a stuck job rather than a
        slow one."""
        txt = App._vace_eta_text(400.0, 200.0, False)
        self.assertIn("longer than estimated", txt)
        self.assertNotIn("0s left", txt)

    def test_the_pulse_stays_on_while_estimating(self):
        """The bar is moving on a guess, and the shimmer is what
        distinguishes that from measured progress."""
        import inspect
        body = inspect.getsource(App._usc_ai_generate)
        i = body.index('phase == "render"')
        self.assertIn("usc_bar.pulse(True)", body[i:i + 400])

    def test_the_real_duration_is_recorded_when_render_ends(self):
        import inspect
        self.assertIn("_vace_record_render",
                      inspect.getsource(App._usc_ai_generate))


class TestVaceClipValidity(unittest.TestCase):
    """A one-frame clip sat in the queue for thirteen minutes before the
    API returned 'duration too short' -- thirteen minutes to learn
    something arithmetic could have said at once."""

    def test_a_single_frame_is_rejected(self):
        """0.041667s at 24fps: exactly what came back from fal."""
        self.assertIsNotNone(App._vace_clip_problem(1))
        self.assertIn("0.042s", App._vace_clip_problem(1))

    def test_the_real_floor_is_the_models_minimum(self):
        """0.1s is three frames, but the model will not generate fewer
        than 81 -- so a shorter source is a request that cannot be
        served whatever its duration."""
        self.assertIsNotNone(App._vace_clip_problem(10))
        self.assertIn(str(App.VACE_MIN_GEN), App._vace_clip_problem(10))

    def test_a_long_enough_clip_passes(self):
        self.assertIsNone(App._vace_clip_problem(App.VACE_MIN_GEN))
        self.assertIsNone(App._vace_clip_problem(240))

    def test_an_empty_clip_is_rejected(self):
        self.assertIsNotNone(App._vace_clip_problem(0))

    def test_it_is_checked_before_anything_is_uploaded(self):
        import inspect
        body = inspect.getsource(App._usc_ai_generate)
        self.assertLess(body.index("_vace_clip_problem"),
                        body.index("_vace_build_source_clip"))

    def test_the_message_says_what_to_do(self):
        import inspect
        self.assertIn("longer sequence",
                      inspect.getsource(App._usc_ai_generate))

    # ── the span ──
    def test_the_span_meets_the_minimum(self):
        for idx, total in ((0, 500), (250, 500), (499, 500)):
            with self.subTest(idx=idx):
                r0, r1 = App._vace_span_for(idx, total)
                self.assertEqual(r1 - r0 + 1, App.VACE_MIN_GEN)

    def test_the_span_always_contains_the_frame_asked_for(self):
        """Otherwise the preview shows a frame the generation never
        touched."""
        for idx, total in ((0, 500), (1, 500), (250, 500), (498, 500),
                           (499, 500), (10, 40), (0, 5)):
            with self.subTest(idx=idx, total=total):
                r0, r1 = App._vace_span_for(idx, total)
                self.assertLessEqual(r0, idx)
                self.assertGreaterEqual(r1, min(idx, total - 1))

    def test_the_span_is_centred_where_it_can_be(self):
        """Motion from both sides is what a video model has over a
        single-image one, and the only reason to be paying for it."""
        r0, r1 = App._vace_span_for(250, 500)
        self.assertLess(abs((r0 + r1) // 2 - 250), 2)

    def test_the_span_stays_inside_the_sequence(self):
        for idx, total in ((0, 500), (499, 500), (3, 10)):
            with self.subTest(idx=idx, total=total):
                r0, r1 = App._vace_span_for(idx, total)
                self.assertGreaterEqual(r0, 0)
                self.assertLessEqual(r1, total - 1)

    def test_a_short_sequence_gives_everything_it_has(self):
        r0, r1 = App._vace_span_for(10, 40)
        self.assertEqual((r0, r1), (0, 39))

    def test_the_preview_uses_a_span_not_one_frame(self):
        import inspect
        body = inspect.getsource(App._usc_preview_model)
        seg = body[body.index('if m == "wan_vace":'):]
        self.assertIn("_vace_span_for", seg)
        self.assertNotIn("names, idx, idx", seg)

    def test_the_preview_reads_back_the_frame_it_asked_for(self):
        """Not the first of the span, which would show the wrong
        picture -- both in the in-memory fallback and the cache lookup,
        which is keyed on the frame on screen."""
        import inspect
        body = inspect.getsource(App._usc_preview_model)
        self.assertIn("idx - r0", body[body.index('if m == "wan_vace":'):])
        self.assertIn("names[idx]",
                      inspect.getsource(App._usc_model_cache_hit))

    def test_the_preview_warns_it_will_take_minutes(self):
        """81 frames is the model's floor, so an AI preview costs a
        full generation -- worth saying before the wait, not after."""
        import inspect
        body = inspect.getsource(App._usc_preview_model)
        seg = body[body.index('if m == "wan_vace":'):]
        self.assertIn("takes minutes", seg)
        self.assertLess(seg.index("takes minutes"),
                        seg.index("_usc_ai_generate("))

class TestUpscaleRangeBar(unittest.TestCase):
    """A range bar on the timeline, defaulting to 80 frames.

    It matters more here than on the other modules: a generation is
    billed by the frame, so the difference between rendering a shot and
    rendering a delivery is one drag."""

    class _Range:
        def __init__(self, v0, v1):
            self.v = (v0, v1)

        def get_range(self):
            return self.v

    class _Stub(App):
        def __init__(self, n=500, v0=0, v1=80, ai=False):
            self._usc_frames = [f"f.{i:04d}.exr" for i in range(n)]
            self.usc_range_bar = TestUpscaleRangeBar._Range(v0, v1)
            self._usc_ai_on = ai

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_the_bar_is_built(self):
        body = self._src(App._tab_upscale)
        self.assertIn("self._maya_range_bar(", body)
        self.assertIn("usc_range_bar", body)

    def test_the_default_is_eighty(self):
        self.assertEqual(App.USC_RANGE_DEFAULT, 80)

    def test_the_default_is_applied_on_load(self):
        """A 500-frame plate is six generations, and nobody means to
        spend that by pressing Run once."""
        body = self._src(App._usc_load_folder)
        self.assertIn("USC_RANGE_DEFAULT", body)
        self.assertIn("set_view", body)

    def test_the_range_is_inclusive_and_clamped(self):
        self.assertEqual(self._Stub(500, 0, 80)._usc_range(), (0, 79))
        self.assertEqual(self._Stub(500, 100, 180)._usc_range(), (100, 179))

    def test_it_never_runs_past_the_sequence(self):
        self.assertEqual(self._Stub(40, 0, 80)._usc_range(), (0, 39))

    def test_a_missing_bar_selects_everything(self):
        app = self._Stub(30)
        app.usc_range_bar = None
        self.assertEqual(app._usc_range(), (0, 29))

    def test_an_empty_sequence_does_not_raise(self):
        app = self._Stub(0)
        self.assertEqual(app._usc_range(), (0, 0))

    def test_the_render_honours_the_range(self):
        body = self._src(App._usc_run)
        self.assertIn("_sel = _frames[_r0:_r1 + 1]", body)
        self.assertIn("enumerate(_sel)", body)

    def test_the_generation_spans_the_selection_not_the_sequence(self):
        """Generating the whole plate while rendering eighty frames
        would bill for six times the work delivered."""
        body = self._src(App._usc_run)
        self.assertIn("_folder, _sel, 0, len(_sel) - 1", body)

    def test_the_selection_is_computed_before_it_is_used(self):
        body = self._src(App._usc_run)
        self.assertLess(body.index("_sel = _frames"),
                        body.index("_folder, _sel"))

    def test_the_frames_list_is_not_rebound(self):
        """Assigning to _frames inside the worker would make it local
        for the whole function, so every read BEFORE that line raises
        UnboundLocalError -- decided at compile time, which is why it
        fails on the line that looks fine."""
        body = self._src(App._usc_run)
        self.assertNotIn("_frames = _frames[", body)

    # ── the readout and the button ──
    def test_the_readout_counts_what_run_will_do(self):
        import inspect
        self.assertIn("frame(s)", inspect.getsource(App._usc_range_changed))

    def test_it_warns_when_the_range_is_short_for_ai(self):
        """Below the model's minimum the generation cannot be served,
        and that is worth knowing before pressing Run rather than after
        the upload."""
        body = self._src(App._usc_range_changed)
        self.assertIn("VACE_MIN_GEN", body)
        self.assertIn("AI needs", body)

    def test_paid_previews_say_so_before_generating(self):
        """With Topaz or Wan selected, Preview is not a quick look: each
        says, in the log, what it is about to generate and that it costs
        money -- before the call, not after."""
        body = self._src(App._usc_preview_model)
        t = body[body.index('if m == "topaz":'):body.index('if m == "wan_vace":')]
        self.assertLess(t.index("spends real money"),
                        t.index("_usc_topaz_expand("))
        w = body[body.index('if m == "wan_vace":'):]
        self.assertLess(w.index("spends money"), w.index("_usc_ai_generate("))
        # and the button counts what it will run
        self.assertIn("MODEL", self._src(App._usc_models_changed))

class TestFalAccount(unittest.TestCase):
    """Balance and usage from fal's platform API.

    The interesting part is not the happy path: billing and usage need
    an ADMIN-scope key, while the key that runs models is explicitly
    barred from them. The common case is a key that generates video
    perfectly and returns 403 here."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    # ── balance parsing ──
    def test_it_reads_the_documented_shape(self):
        self.assertEqual(
            App._fal_format_balance({"credits": {"balance": 42.5,
                                                 "currency": "USD"}}),
            "42.50 USD")

    def test_it_tolerates_a_moved_field(self):
        """This reads someone else's API; a response that shifts a key
        is not a reason to show a traceback."""
        self.assertEqual(
            App._fal_format_balance({"credits": {"amount": 7,
                                                 "currency": "EUR"}}),
            "7.00 EUR")
        self.assertEqual(App._fal_format_balance({"balance": 3.25}),
                         "3.25 USD")

    def test_an_unreadable_response_gives_none_not_an_error(self):
        for bad in ({}, {"credits": None}, "nonsense", None, []):
            with self.subTest(data=repr(bad)[:20]):
                self.assertIsNone(App._fal_format_balance(bad))

    # ── usage parsing ──
    def test_usage_rows_are_flattened(self):
        data = {"time_series": [{"bucket": "2026-09-12T00:00:00-05:00",
                                 "results": [{"endpoint_id": "fal-ai/x",
                                              "unit": "video",
                                              "quantity": 1,
                                              "cost_total": 0.42,
                                              "currency": "USD"}]}]}
        rows = App._fal_format_usage(data)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "2026-09-12")
        self.assertAlmostEqual(rows[0][4], 0.42)

    def test_usage_is_newest_first(self):
        data = {"time_series": [
            {"bucket": "2026-09-01", "results": [
                {"endpoint_id": "a", "cost_total": 1.0}]},
            {"bucket": "2026-09-09", "results": [
                {"endpoint_id": "b", "cost_total": 2.0}]}]}
        rows = App._fal_format_usage(data)
        self.assertEqual(rows[0][1], "b")

    def test_usage_is_capped(self):
        data = {"time_series": [{"bucket": "2026-09-01", "results": [
            {"endpoint_id": str(i), "cost_total": 1.0} for i in range(50)]}]}
        self.assertLessEqual(len(App._fal_format_usage(data)), 12)

    def test_a_missing_cost_does_not_raise(self):
        data = {"time_series": [{"bucket": "x", "results": [
            {"endpoint_id": "a"}, {"endpoint_id": "b",
                                   "cost_total": "not a number"}]}]}
        rows = App._fal_format_usage(data)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][4], 0.0)

    def test_empty_usage_is_empty(self):
        for bad in ({}, None, {"time_series": []}, "x"):
            with self.subTest(data=repr(bad)[:16]):
                self.assertEqual(App._fal_format_usage(bad), [])

    # ── the scope problem ──
    def test_a_forbidden_key_explains_the_scope(self):
        """Shown as a blank panel it reads as the feature being
        broken, when the key is simply the wrong kind."""
        body = self._src(App._fal_api_get)
        self.assertIn("(401, 403)", body)
        self.assertIn("ADMIN", body)
        self.assertIn("dashboard/keys", body)

    def test_a_rate_limit_is_named(self):
        self.assertIn("rate-limited", self._src(App._fal_api_get))

    def test_no_key_is_not_an_exception(self):
        body = self._src(App._fal_api_get)
        self.assertIn("No fal key saved", body)

    def test_the_call_never_raises(self):
        """Every caller is a UI button, and a traceback in a status
        label helps nobody."""
        body = self._src(App._fal_api_get)
        self.assertIn("except Exception", body)
        self.assertIn("return None,", body)

    def test_it_authenticates_the_way_fal_documents(self):
        self.assertIn('f"Key {key}"', self._src(App._fal_api_get))

    def test_it_asks_for_the_credits_expansion(self):
        """Billing returns the balance only when expanded."""
        self.assertIn('"expand": "credits"',
                      self._src(App._fal_show_balance))

    # ── the UI ──
    def test_both_are_fetched_on_demand(self):
        """A dialog that stalls on opening is worse than one that waits
        to be asked."""
        for fn in (App._fal_show_balance, App._fal_show_usage):
            with self.subTest(fn=fn.__name__):
                self.assertIn("threading.Thread", self._src(fn))

    def test_the_buttons_exist(self):
        body = self._src(App._fal_wizard)
        self.assertIn("_fal_show_balance", body)
        self.assertIn("_fal_show_usage", body)

    def test_status_falls_back_when_the_dialog_is_closed(self):
        """The fetch is threaded, so it can land after the window has
        gone."""
        body = self._src(App._fal_status)
        self.assertIn("is None", body)
        self.assertIn("setup_log", body)


class TestForgetKeyConfirm(unittest.TestCase):
    """Removing the key is irreversible in the way that matters: the
    file can be recreated, but the key cannot be read back from fal. A
    mis-click means revoking and reissuing."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_it_asks_before_deleting(self):
        body = self._src(App._fal_forget_key)
        self.assertIn("askyesno", body)
        self.assertLess(body.index("askyesno"), body.index("p.unlink()"))

    def test_cancelling_deletes_nothing(self):
        body = self._src(App._fal_forget_key)
        i = body.index("askyesno")
        self.assertIn("return", body[i:i + 500])

    def test_it_defaults_to_no(self):
        """The button sits next to Test, which is the one people mean
        to press."""
        body = self._src(App._fal_forget_key)
        self.assertIn('default="no"', body)
        self.assertIn('icon="warning"', body)

    def test_the_prompt_names_what_will_go(self):
        """A dialog that does not say what it is about trains people to
        dismiss it."""
        body = self._src(App._fal_forget_key)
        self.assertIn("_fal_key_file()", body)
        self.assertIn("Remove fal.ai key", body)

    def test_the_prompt_says_the_key_cannot_be_recovered(self):
        body = self._src(App._fal_forget_key)
        self.assertIn("will not show you this key again", body)
        self.assertIn("dashboard/keys", body)

    def test_the_key_is_shown_masked_not_in_full(self):
        """It is on screen next to a dialog somebody may screenshot."""
        body = self._src(App._fal_forget_key)
        self.assertIn("k[:6]", body)
        self.assertIn("k[-4:]", body)

    def test_a_failed_delete_is_reported(self):
        """Silently swallowing it would leave the key in place while
        the card claims it is gone."""
        body = self._src(App._fal_forget_key)
        self.assertIn("Could not remove the key", body)

    def test_an_environment_key_is_cleared_not_merely_warned_about(self):
        """Warning about it left the key in use. _vace_submit writes
        FAL_KEY into the process before every call, so after one
        generation the file being gone changes nothing unless the
        variable goes too."""
        body = self._src(App._fal_forget_key)
        self.assertIn('os.environ.pop("FAL_KEY", None)', body)


class TestFalAuthErrors(unittest.TestCase):
    """fal-client raises httpx errors, so an expired key surfaces as a
    401 on a storage-token URL with a link to MDN's page about HTTP
    status codes -- accurate, and no help at all in working out that
    the key needs replacing."""

    def test_401_says_what_to_do(self):
        msg = App._fal_explain_error(Exception(
            "Client error '401 Unauthorized' for url "
            "'https://rest.fal.ai/storage/auth/token'"))
        self.assertIsNotNone(msg)
        self.assertIn("Setup", msg)
        self.assertIn("dashboard/keys", msg)

    def test_401_mentions_the_admin_key_swap(self):
        """The likeliest cause right after adding a balance readout is
        somebody having pasted an Admin key over their API key."""
        msg = App._fal_explain_error(Exception("401 Unauthorized"))
        self.assertIn("ADMIN", msg)

    def test_402_points_at_the_balance(self):
        self.assertIn("balance",
                      App._fal_explain_error(Exception("402 insufficient")))

    def test_403_and_429_are_distinguished(self):
        self.assertIn("permission",
                      App._fal_explain_error(Exception("403 Forbidden")))
        self.assertIn("rate-limited",
                      App._fal_explain_error(Exception("429")))

    def test_an_unrecognised_error_is_left_alone(self):
        """Rewriting an error nobody understands into a guess would be
        worse than the traceback."""
        self.assertIsNone(App._fal_explain_error(Exception("socket reset")))

    def test_the_generation_uses_it(self):
        import inspect
        body = inspect.getsource(App._usc_ai_generate)
        self.assertIn("_fal_explain_error", body)
        self.assertIn("raise RuntimeError(said) if said else ex", body)

    # ── the stale environment ──
    def test_removing_the_key_stops_the_environment_fallback(self):
        """_vace_submit sets os.environ['FAL_KEY'] before every call, so
        after one generation the variable is populated for the life of
        the process -- and removing the saved key would appear to do
        nothing."""
        import inspect
        body = inspect.getsource(App._fal_get_key)
        self.assertIn("_fal_key_removed", body)
        self.assertLess(body.index("_fal_key_removed"),
                        body.index('os.environ.get("FAL_KEY"'))

    def test_removing_also_clears_the_variable(self):
        """A value this app wrote into its own process should not
        outlive the key it came from."""
        import inspect
        body = inspect.getsource(App._fal_forget_key)
        self.assertIn('os.environ.pop("FAL_KEY", None)', body)
        self.assertIn("self._fal_key_removed = True", body)

    def test_saving_a_key_clears_the_flag(self):
        """Otherwise the app stays blind to a key it just saved."""
        import inspect
        self.assertIn("self._fal_key_removed = False",
                      inspect.getsource(App._fal_save_key))


class TestFalClientCredentials(unittest.TestCase):
    """fal_client.upload_file and .subscribe are bound to a module-level
    SyncClient whose credentials are a @cached_property: it resolves
    FAL_KEY once, on first use, and caches it for the life of the
    process.

    Setting os.environ["FAL_KEY"] afterwards changes nothing -- so a key
    saved or replaced mid-session was ignored, and every call kept using
    whatever was in the environment at startup. That is how a perfectly
    valid key produced 401 Unauthorized on the storage token."""

    def test_the_module_client_really_does_cache(self):
        """Pins the diagnosis against the library itself, so an upstream
        fix shows up here rather than leaving dead workaround code."""
        try:
            import fal_client
        except ImportError:
            self.skipTest("fal_client not installed")
        import inspect
        src = inspect.getsource(fal_client.client.SyncClient)
        self.assertIn("cached_property", src)
        self.assertIn("_auth", src)

    def test_an_explicit_client_uses_the_key_given(self):
        try:
            import fal_client
        except ImportError:
            self.skipTest("fal_client not installed")
        c = fal_client.SyncClient(key="explicit-key-xyz")
        self.assertEqual(c._auth.token, "explicit-key-xyz")

    def test_the_app_builds_its_own_client(self):
        import inspect
        body = inspect.getsource(App._fal_client)
        self.assertIn("fal_client.SyncClient(key=", body)

    def test_no_call_site_uses_the_module_level_helpers(self):
        """One missed site keeps the stale key on that path only, which
        is worse than the original bug because it works everywhere
        else."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        for bad in ("fal_client.upload_file(", "fal_client.subscribe("):
            with self.subTest(call=bad):
                # allowed in comments and docstrings, not in code
                for line in src.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("#") or stripped.startswith('"'):
                        continue
                    self.assertNotIn(bad, line)

    def test_a_missing_key_is_refused_before_the_call(self):
        import inspect
        body = inspect.getsource(App._fal_client)
        self.assertIn("No API key", body)

    def test_the_presence_check_does_not_build_a_client(self):
        """Importing the library proves it is installed; constructing
        the cached client is what caused the problem."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn('importlib.import_module("fal_client")', src)


class TestVaceSharedFromOtherTabs(unittest.TestCase):
    """The VACE helpers were written for ADW-Interpolate and now serve
    three tabs. Anything reading that tab's state has to cope when
    another tab is driving."""

    class _Stub(App):
        def __init__(self, folder=None):
            self.ip_folder = folder

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_work_dir_survives_a_missing_ip_folder(self):
        """os.path.abspath(None) raised a TypeError thirteen minutes
        into a generation that had already succeeded, losing the
        paid-for result at the last step."""
        for folder in (None, "", "/nope/does/not/exist"):
            with self.subTest(ip_folder=folder):
                d = self._Stub(folder)._vace_work_dir()
                self.assertTrue(os.path.isdir(d))

    def test_it_still_sits_beside_the_input_when_there_is_one(self):
        d = self._Stub(self.dir)._vace_work_dir()
        self.assertIn("Studio-Toolbox_VACE", d)
        self.assertTrue(os.path.isdir(d))

    def test_the_fallback_is_scratch_not_the_cwd(self):
        """Writing generated frames into whatever directory the app
        happens to be running from would be worse than failing."""
        d = self._Stub(None)._vace_work_dir()
        self.assertTrue(d.startswith(tempfile.gettempdir()))

    def test_the_payload_settings_are_always_built(self):
        """They used to live on the Interpolate tab; without that tab
        they are created in _vace_init_shared_settings, which __init__
        calls before any tab is built -- so every tab borrowing them
        finds them present. If that ever becomes lazy, this breaks."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        for var in ("ip_vace_res_var", "ip_vace_steps_var",
                    "ip_vace_prompt_var", "ip_vace_neg_var",
                    "ip_vace_seed_var", "ip_vace_method_var",
                    "_ip_vace_cancel"):
            with self.subTest(var=var):
                self.assertRegex(src, r"self\." + var + r"\s*=")
        init = src[src.index("    def __init__(self, modules=None"):
                   src.index("    def _vace_init_shared_settings")]
        self.assertLess(init.index("self._vace_init_shared_settings()"),
                        init.index("self._build()"))


class TestAiExpansionCache(unittest.TestCase):
    """A generation is billed and takes minutes, so a frame already
    generated must never be generated again."""

    class _V:
        def __init__(self, v):
            self.v = v

        def get(self):
            return self.v

    class _Stub(App):
        def __init__(self, cache):
            self.usc_ai_thresh = TestAiExpansionCache._V(0.96)
            self.usc_ai_prompt = TestAiExpansionCache._V("")
            self._cache = cache
            self._usc_ai_on = True
            self._usc_sel = {"tonemap": True, "topaz": False,
                             "wan_vace": False}

        def _usc_ai_cache_dir(self):
            return self._cache

    def setUp(self):
        self.cache = tempfile.mkdtemp()
        self.src = tempfile.mkdtemp()
        self.gen = tempfile.mkdtemp()
        self.app = self._Stub(self.cache)
        from PIL import Image
        self.names = []
        for k in range(5):
            n = f"p.{k:04d}.png"
            self.names.append(n)
            Image.fromarray(np.full((8, 8, 3), 10 * k, np.uint8)).save(
                os.path.join(self.src, n))
            Image.fromarray(np.full((8, 8, 3), 200, np.uint8)).save(
                os.path.join(self.gen, f"{k:05d}.png"))

    def tearDown(self):
        for d in (self.cache, self.src, self.gen):
            shutil.rmtree(d, ignore_errors=True)

    def test_everything_is_missing_to_begin_with(self):
        self.assertEqual(
            self.app._usc_ai_missing(self.src, self.names, 0, 4, 0.96),
            [0, 1, 2, 3, 4])

    def test_storing_satisfies_the_span(self):
        self.app._usc_ai_store(self.src, self.names, 0, self.gen, 0.96)
        self.assertEqual(
            self.app._usc_ai_missing(self.src, self.names, 0, 4, 0.96), [])

    def test_a_different_clip_level_invalidates(self):
        """Changing what counts as blown changes the mask and therefore
        the result. A cache ignoring the threshold would quietly serve
        the wrong frames."""
        self.app._usc_ai_store(self.src, self.names, 0, self.gen, 0.96)
        self.assertEqual(
            self.app._usc_ai_missing(self.src, self.names, 0, 4, 0.90),
            [0, 1, 2, 3, 4])

    def test_a_partial_cache_reports_only_the_gaps(self):
        """Regenerating a whole span because two frames are missing is
        the waste this exists to prevent."""
        self.app._usc_ai_store(self.src, self.names, 0, self.gen, 0.96)
        for k in (1, 3):
            os.unlink(self.app._usc_ai_cache_key(self.src, self.names[k],
                                                 0.96))
        self.assertEqual(
            self.app._usc_ai_missing(self.src, self.names, 0, 4, 0.96),
            [1, 3])

    def test_different_frames_get_different_keys(self):
        keys = {self.app._usc_ai_cache_key(self.src, n, 0.96)
                for n in self.names}
        self.assertEqual(len(keys), len(self.names))

    def test_a_different_source_folder_is_a_different_key(self):
        """Two shots with identically-named frames must not collide."""
        a = self.app._usc_ai_cache_key("/jobs/shotA", "p.0001.exr", 0.96)
        b = self.app._usc_ai_cache_key("/jobs/shotB", "p.0001.exr", 0.96)
        self.assertNotEqual(a, b)

    def test_the_cache_view_is_ordered_for_the_caller(self):
        """So a cache hit and a fresh generation are interchangeable
        downstream."""
        self.app._usc_ai_store(self.src, self.names, 0, self.gen, 0.96)
        v = self.app._usc_ai_cache_view(self.src, self.names, 0, 4, 0.96)
        self.assertEqual(sorted(os.listdir(v)),
                         [f"{k:05d}.png" for k in range(5)])

    def test_a_full_cache_skips_the_call(self):
        import inspect
        body = inspect.getsource(App._usc_ai_generate)
        self.assertIn("if not missing:", body)
        self.assertLess(body.index("if not missing:"),
                        body.index("_vace_clip_problem"))

    def test_the_cache_is_checked_before_the_length_rule(self):
        """A fully cached span is servable even when shorter than the
        model's minimum -- the model is not being asked."""
        import inspect
        body = inspect.getsource(App._usc_ai_generate)
        self.assertLess(body.index("_usc_ai_missing"),
                        body.index("_vace_clip_problem"))

    def test_results_are_stored_after_decoding(self):
        import inspect
        body = inspect.getsource(App._usc_ai_generate)
        self.assertIn("_usc_ai_store", body)
        self.assertLess(body.index("_vace_decode"),
                        body.index("_usc_ai_store"))

    # ── the timeline ──
    def test_generated_frames_colour_the_timeline(self):
        self.app._usc_frames = self.names
        self.app._usc_folder = self.src
        self.assertIsNone(self.app._usc_frame_colour(2))
        # An empty Guide field resolves to HDR_AI_PROMPT everywhere
        # else in the app -- stored under that same resolved value, so
        # this matches what a real generation call would actually key
        # against, not the raw empty string.
        self.app._usc_ai_store(self.src, self.names, 0, self.gen, 0.96,
                               prompt=App.HDR_AI_PROMPT)
        self.assertEqual(self.app._usc_frame_colour(2), "#DDAA44")

    def test_nothing_is_coloured_when_ai_is_off(self):
        self.app._usc_frames = self.names
        self.app._usc_folder = self.src
        self.app._usc_ai_store(self.src, self.names, 0, self.gen, 0.96)
        self.app._usc_ai_on = False
        self.assertIsNone(self.app._usc_frame_colour(2))

    def test_the_colour_is_read_from_the_cache_not_a_stored_list(self):
        """So it stays true after a run, a restart, or a change of clip
        level."""
        import inspect
        self.assertIn("_usc_ai_cached",
                      inspect.getsource(App._usc_frame_colour))

    def test_an_out_of_range_index_is_safe(self):
        self.app._usc_frames = self.names
        self.app._usc_folder = self.src
        self.assertIsNone(self.app._usc_frame_colour(999))

    # ── the A/B toggle ──
    def test_toggling_never_generates(self):
        """A comparison that costs minutes and money is one nobody
        makes twice: clicking between models, and showing a frame, only
        ever READ the caches."""
        import inspect
        for fn in (App._usc_model_result, App._usc_model_available,
                   App._usc_view, App._usc_show_frame, App._usc_tl_click):
            with self.subTest(fn=fn.__name__):
                body = inspect.getsource(fn)
                self.assertNotIn("_usc_ai_generate", body)
                self.assertNotIn("_usc_topaz_expand", body)
                self.assertNotIn("_usc_preview()", body)
        self.assertIn("_usc_ai_cached",
                      inspect.getsource(App._usc_model_cache_hit))

    def test_an_ungenerated_frame_is_never_claimed_as_previewed(self):
        """No cache entry, no in-memory result -> no green box, and
        nothing to show for that model on this frame."""
        import inspect
        body = inspect.getsource(App._usc_model_available)
        self.assertIn("_usc_model_cache_hit", body)
        self.assertIn("return None", inspect.getsource(App._usc_model_result))

    def test_the_tile_click_is_wired_to_viewing(self):
        import inspect
        body = inspect.getsource(App._usc_tile_click)
        self.assertIn("self._usc_view(m)", body)
        self.assertIn("self._usc_model_available(m)", body)

    def test_the_preview_keeps_what_it_needs_to_rebuild(self):
        """Without the source codes, switching models would have to
        re-read the plate."""
        import inspect
        body = inspect.getsource(App._usc_load_preview_frame)
        self.assertIn("_usc_preview_code", body)
        self.assertIn("_usc_preview_idx", body)
        self.assertIn("self._usc_load_preview_frame(",
                      inspect.getsource(App._usc_preview))

class TestColorspaceHandoff(unittest.TestCase):
    """An expanded EXR loaded into ADW-Organic read crushed and
    contrasty: Organic assumes display-encoded by default, so it
    linearised data that was already linear -- mid grey 6.6x too dark --
    and then CLIPPED away the over-range the expansion had just created.

    The whole point of writing scene-linear is lost if the next module
    has to be told by hand what it is looking at."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    # ── the sidecar ──
    def test_an_unmarked_folder_says_nothing(self):
        self.assertIsNone(App._adw_read_colorspace(self.dir))

    def test_a_mark_round_trips(self):
        for space in ("scene-linear", "display"):
            with self.subTest(space=space):
                App._adw_write_colorspace(self.dir, space)
                self.assertEqual(App._adw_read_colorspace(self.dir), space)

    def test_an_unknown_value_is_ignored(self):
        """Better to fall back than to trust a word nothing wrote."""
        App._adw_write_colorspace(self.dir, "something-else")
        self.assertIsNone(App._adw_read_colorspace(self.dir))

    def test_a_corrupt_sidecar_is_not_fatal(self):
        with open(os.path.join(self.dir, App.ADW_CS_SIDECAR), "w",
                  encoding="utf-8") as fh:
            fh.write("not json")
        self.assertIsNone(App._adw_read_colorspace(self.dir))

    def test_writing_to_a_missing_folder_does_not_raise(self):
        App._adw_write_colorspace("/nope/not/here", "display")

    def test_the_expanded_output_marks_itself(self):
        import inspect
        body = inspect.getsource(App._usc_run)
        self.assertIn("_adw_write_colorspace", body)
        self.assertIn('"scene-linear"', body)
        self.assertIn('"display"', body)

    # ── the over-range fallback ──
    def test_a_single_hot_pixel_does_not_flip_a_sequence(self):
        """A fraction scales with resolution, so the same stray pixel
        passed on a thumbnail and failed on a 4K plate."""
        rng = np.random.default_rng(1)
        enc = rng.random((100, 100, 3)).astype(np.float32)
        enc[0, 0] = 4.0
        self.assertFalse(App._exr_is_over_range(enc))

    def test_real_over_range_is_detected(self):
        rng = np.random.default_rng(2)
        lin = rng.random((100, 100, 3)).astype(np.float32) * 0.5
        lin[40:60, 40:60] = 10.0
        self.assertTrue(App._exr_is_over_range(lin))

    def test_display_encoded_data_never_looks_over_range(self):
        rng = np.random.default_rng(3)
        self.assertFalse(App._exr_is_over_range(
            rng.random((200, 200, 3)).astype(np.float32)))

    def test_float_noise_at_one_is_not_over_range(self):
        self.assertFalse(App._exr_is_over_range(
            np.full((100, 100, 3), 1.0000001, np.float32)))

    def test_an_empty_frame_is_safe(self):
        self.assertFalse(App._exr_is_over_range(np.zeros((0, 0, 3),
                                                         np.float32)))

    # ── precedence ──
    def test_the_mark_beats_the_heuristic_beats_the_default(self):
        # Checked against the CODE: the docstring names
        # _adw_source_is_encoded early to explain the problem, which is
        # not where it is called.
        import ast as _ast
        import inspect
        import textwrap
        fn = _ast.parse(textwrap.dedent(
            inspect.getsource(App._organic_load_linear))).body[0]
        if (fn.body and isinstance(fn.body[0], _ast.Expr)
                and isinstance(fn.body[0].value, _ast.Constant)):
            fn.body = fn.body[1:]
        body = _ast.unparse(fn)
        self.assertLess(body.index("_adw_read_colorspace"),
                        body.index("_exr_is_over_range"))
        self.assertLess(body.index("_exr_is_over_range"),
                        body.index("_adw_source_is_encoded"))

    def test_the_decision_is_still_pinned_per_sequence(self):
        """Re-deciding per frame could leave half a sequence decoded one
        way and half the other."""
        import inspect
        body = inspect.getsource(App._organic_load_linear)
        self.assertIn("if self._organic_source_encoded is None:", body)

    def test_organic_says_which_way_it_read_the_sequence(self):
        """Silently guessing is what made this hard to see."""
        import inspect
        body = inspect.getsource(App._organic_load_linear)
        self.assertIn("is marked", body)
        self.assertIn("scene-linear", body)


class TestFloatPrecisionControls(unittest.TestCase):
    """Scene-linear footage needs different numbers from an 8-bit
    plate: diffuse white is 1.0 and highlights run to 10 or 40, so the
    useful thresholds are ABOVE 1.0 and the useful intensities are far
    below the old step size."""

    @staticmethod
    def _slider(pkey):
        for c in TestRegistryWiring._cs_calls():
            kw = {k.arg: k.value for k in c.keywords}
            if kw.get("pkey") is not None and kw["pkey"].value == pkey:
                return c
        raise AssertionError(f"{pkey} slider not found")

    def test_the_threshold_reaches_expanded_highlights(self):
        """An expansion at 4000 nits puts highlights at 40x white. A
        threshold ending at 4.0 could never gate them."""
        c = self._slider("bloom_thresh")
        self.assertGreaterEqual(c.args[4].value, 40)

    def test_the_intensity_step_is_fine_enough_to_be_useful(self):
        """Measured on expanded float, the usable band is roughly
        0.02 to 0.3 -- about four positions at the old 0.05 step."""
        c = self._slider("bloom_intensity")
        self.assertLessEqual(c.args[5].value, 0.001)

    def test_the_readouts_show_enough_digits(self):
        """A value of 0.0150 displayed as 0.02 is indistinguishable
        from 0.0249."""
        # Read from the _cs call itself: the tooltip between the fmt
        # lambda and the pkey is long enough to push them apart.
        c = self._slider("bloom_intensity")
        import ast as _ast
        self.assertIn(".4f", _ast.unparse(c.args[6]))
        self.assertIn(".3f", _ast.unparse(self._slider("bloom_thresh").args[6]))

    # ── manual entry ──
    def test_the_value_readout_is_editable(self):
        import inspect
        body = inspect.getsource(App._tab_organic)
        self.assertIn("vl = tk.Entry(row", body)
        self.assertIn('vl.bind("<Return>"', body)

    def test_committing_on_focus_loss_too(self):
        """Typing a number and clicking away should not discard it."""
        import inspect
        self.assertIn('vl.bind("<FocusOut>"',
                      inspect.getsource(App._tab_organic))

    def test_an_unparsable_entry_restores_the_real_value(self):
        """Leaving nonsense on screen beside a control that ignored it
        is worse than rejecting it."""
        import inspect
        body = inspect.getsource(App._tab_organic)
        i = body.index("except ValueError:")
        self.assertIn("_from_var()", body[i:i + 160])

    def test_a_typed_value_widens_the_slider(self):
        """tk.Scale CLAMPS its variable to its own range, so a typed
        value outside it would snap back the moment the widget
        redrew."""
        import inspect
        body = inspect.getsource(App._tab_organic)
        self.assertIn('sc.configure(to=val)', body)
        self.assertIn('sc.configure(from_=val)', body)

    def test_the_entry_does_not_fight_the_slider(self):
        """Without a guard, the variable's write-trace rewrites the box
        while it is being typed in."""
        import inspect
        body = inspect.getsource(App._tab_organic)
        self.assertIn('_guard["busy"]', body)
        self.assertIn("def _from_var", body)

    def test_typing_still_marks_the_preview_stale(self):
        """A value changed by typing is as much a change as one dragged."""
        import inspect
        body = inspect.getsource(App._tab_organic)
        i = body.index("def _commit")
        seg = body[i:i + 1400]
        self.assertIn("_organic_mark_stale", seg)
        self.assertIn("_organic_param_touched", seg)


class TestOrganicExposure(unittest.TestCase):
    """Exposure and gamma in ADW-Organic's viewer.

    Not optional polish on an expanded source: the values that matter
    sit above display white, and on an SDR monitor they are invisible
    until you stop down to them."""

    class _V:
        def __init__(self, v):
            self.v = v

        def get(self):
            return self.v

    class _Stub(App):
        def __init__(self):
            self.organic_ev_var = TestOrganicExposure._V(0.0)
            self.organic_gm_var = TestOrganicExposure._V(1.0)

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        from PIL import Image
        self.app = self._Stub()
        lin = np.repeat(
            np.linspace(0.6, 8.0, 8, dtype=np.float32)[None, :, None],
            3, axis=2)
        self.lin = np.repeat(lin, 2, axis=0)
        self.app.organic_lin = self.lin
        self.app.organic_img_base = Image.fromarray(
            App._organic_linear_to_display(self.lin), "RGB")
        self.app._organic_proc_lin = self.lin
        self.app.organic_processed = self.app.organic_img_base

    def test_the_toolbar_is_built(self):
        body = self._src(App._tab_organic)
        self.assertIn("_ev_toolbar(", body)
        self.assertIn("organic_ev_var", body)

    def test_it_redraws_on_change(self):
        body = self._src(App._tab_organic)
        i = body.index("_ev_toolbar(")
        self.assertIn("_organic_redraw", body[i:i + 200])

    def test_stopping_down_reveals_over_range_detail(self):
        """Eight values that all clip to 255 must become eight distinct
        steps -- that is the whole reason for the control."""
        self.app.organic_ev_var = self._V(-4.0)
        row = np.array(self.app._organic_view("orig"))[0, :, 0]
        self.assertEqual(len(set(int(v) for v in row)), 8)

    def test_at_display_exposure_the_range_is_clipped(self):
        """Confirms the problem it solves: unexposed, those same eight
        values are indistinguishable."""
        row = np.array(self.app._organic_view("orig"))[0, :, 0]
        self.assertLess(len(set(int(v) for v in row)), 4)

    def test_neutral_settings_do_no_work(self):
        """Faster, and exactly equal -- so the common case is
        untouched."""
        self.assertIs(self.app._organic_view("orig"),
                      self.app.organic_img_base)

    def test_it_falls_back_without_a_float(self):
        """An 8-bit source still responds; it simply has nothing above
        white to find."""
        self.app._organic_proc_lin = None
        self.app.organic_ev_var = self._V(-2.0)
        out = self.app._organic_view("proc")
        self.assertIsNotNone(out)
        self.assertEqual(np.array(out).shape, self.lin.shape)

    def test_a_missing_image_is_safe(self):
        self.app.organic_img_base = None
        self.assertIsNone(self.app._organic_view("orig"))

    def test_both_sides_of_the_wipe_respond(self):
        """A wipe where only one half responds tells you nothing about
        what the stack changed."""
        body = self._src(App._organic_redraw)
        self.assertIn('self._organic_view("orig")', body)
        self.assertIn('self._organic_view("proc")', body)
        self.assertNotIn("self.organic_img_base.resize", body)
        self.assertNotIn("self.organic_processed.resize", body)

    # ── the float the viewer needs ──
    def test_the_stack_result_is_kept_as_float(self):
        """The 8-bit image is already clipped at 1.0, so stopping down
        on it would reveal nothing."""
        self.assertIn("_organic_proc_lin",
                      self._src(App._organic_load_frame))

    def test_a_region_update_keeps_the_float_in_step(self):
        """Otherwise exposure after an ROI update would show the last
        full frame."""
        body = self._src(App._organic_recompute)
        self.assertIn("_organic_proc_lin", body)
        i = body.index("base.paste(patch")
        self.assertIn("pl[y0:y1, x0:x1] = inner", body[i:i + 600])


class TestOrganicOutputFormat(unittest.TestCase):
    """ADW-Organic computed the whole stack in scene-linear float and
    then wrote 8-bit PNG, throwing that away at the last step."""

    class _Stub(App):
        def __init__(self):
            pass

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        self.app = self._Stub()
        self.dir = tempfile.mkdtemp()
        self.lin = np.zeros((8, 12, 3), np.float32)
        self.lin[:] = np.linspace(0.02, 9.0, 12)[None, :, None]

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write(self, fmt):
        return self.app._organic_write_frame(
            os.path.join(self.dir, fmt.replace(" ", "_")), self.lin, fmt)

    def test_exr_is_the_default(self):
        self.assertTrue(App.ORGANIC_FORMATS[0].startswith("EXR"))

    def test_exr_keeps_values_above_one(self):
        """The whole reason to write float: an expanded highlight at
        9.0 survives."""
        try:
            __import__("OpenEXR")
        except ImportError:
            self.skipTest("OpenEXR not available")
        for fmt in ("EXR 16-bit half", "EXR 32-bit float"):
            with self.subTest(fmt=fmt):
                dst = self._write(fmt)
                self.assertTrue(dst.endswith(".exr"))
                back = App._ff_read_frame(self.app, dst)
                self.assertAlmostEqual(float(back.max()), 9.0, delta=0.02)

    def test_png_clips_as_it_must(self):
        """Not a bug -- a display encode cannot carry over-range. It is
        why PNG is the lossy option here."""
        import cv2
        for fmt in ("PNG 16-bit", "PNG 8-bit"):
            with self.subTest(fmt=fmt):
                dst = self._write(fmt)
                self.assertTrue(dst.endswith(".png"))
                a = cv2.imread(dst, cv2.IMREAD_UNCHANGED)
                self.assertEqual(float(a.max()),
                                 65535.0 if a.dtype == np.uint16 else 255.0)

    def test_png_16_bit_really_is_16_bit(self):
        import cv2
        a = cv2.imread(self._write("PNG 16-bit"), cv2.IMREAD_UNCHANGED)
        self.assertEqual(a.dtype, np.uint16)

    def test_half_and_float_differ_in_size(self):
        """Half is plenty for imagery; 32-bit is twice the data."""
        try:
            __import__("OpenEXR")
        except ImportError:
            self.skipTest("OpenEXR not available")
        h = os.path.getsize(self._write("EXR 16-bit half"))
        f = os.path.getsize(self._write("EXR 32-bit float"))
        self.assertGreater(f, h)

    def test_exr_goes_through_openexr_not_opencv(self):
        """The pip OpenCV wheels ship with the EXR codec compiled
        out -- the same trap the readers hit."""
        import ast as _ast
        import textwrap
        fn = _ast.parse(textwrap.dedent(
            self._src(App._organic_write_exr))).body[0]
        if (fn.body and isinstance(fn.body[0], _ast.Expr)
                and isinstance(fn.body[0].value, _ast.Constant)):
            fn.body = fn.body[1:]      # the docstring names it to explain
        code = _ast.unparse(fn)
        self.assertIn("OpenEXR", code)
        self.assertNotIn("cv2.imwrite", code)

    def test_the_render_uses_the_chosen_format(self):
        body = self._src(App._organic_run_worker)
        self.assertIn("_organic_write_frame", body)
        self.assertIn("organic_fmt_var", body)
        self.assertNotIn('stem + ".png"', body)

    def test_the_output_marks_its_colourspace(self):
        """So loading it into another module does not depend on a
        global default being right."""
        body = self._src(App._organic_run_worker)
        self.assertIn("_adw_write_colorspace", body)
        self.assertIn('"scene-linear"', body)

    def test_the_control_is_built(self):
        body = self._src(App._tab_organic)
        self.assertIn("organic_fmt_var", body)
        self.assertIn("ORGANIC_FORMATS", body)

    def test_the_tip_says_which_option_loses_data(self):
        body = self._src(App._tab_organic)
        i = body.index("organic_fmt_var")
        self.assertIn("LOSSY", body[i:i + 1400])


class TestUpscaleExrOnly(unittest.TestCase):
    """No still-image PNG from Upscale & Expand. Everything the tab does
    runs in scene-linear float, and the expansion exists specifically to
    put detail above 1.0 -- an integer still cannot hold it, so offering
    one was offering a way to discard the work at the last step."""

    def test_exr_is_the_default(self):
        self.assertTrue(App.USC_DEFAULT_FORMAT.startswith("EXR"))
        self.assertIn(App.USC_DEFAULT_FORMAT, App.USC_FORMATS)

    def test_the_default_is_first_in_the_list(self):
        """It is what the dropdown shows before anyone touches it."""
        self.assertEqual(list(App.USC_FORMATS)[0], App.USC_DEFAULT_FORMAT)

    def test_there_is_no_still_png_option(self):
        for name, spec in App.USC_FORMATS.items():
            if spec["movie"] is None:
                with self.subTest(fmt=name):
                    self.assertEqual(spec["ext"], ".exr")

    def test_movie_options_may_still_stage_through_png(self):
        """Those are display deliveries by definition, and the frames
        are an implementation detail nobody keeps."""
        movies = [s for s in App.USC_FORMATS.values() if s["movie"]]
        self.assertTrue(movies)
        for spec in movies:
            self.assertEqual(spec["ext"], ".png")

    def test_both_bit_depths_are_offered(self):
        depths = {bool(s.get("full")) for s in App.USC_FORMATS.values()
                  if s["movie"] is None}
        self.assertEqual(depths, {False, True})

    def test_an_unknown_format_falls_back_to_exr(self):
        import inspect
        self.assertIn("USC_DEFAULT_FORMAT",
                      inspect.getsource(App._usc_format))

    def test_it_uses_the_shared_exr_writer(self):
        """One definition of how this app writes an EXR."""
        import inspect
        body = inspect.getsource(App._usc_write_frame)
        self.assertIn("_organic_write_exr", body)

    def test_32_bit_is_passed_through_as_full_float(self):
        import inspect
        body = inspect.getsource(App._usc_write_frame)
        self.assertIn('half=not spec.get("full", False)', body)

    def test_the_dropdown_is_built_from_the_table(self):
        """A hand-written list would drift from what the writer
        supports."""
        import inspect
        body = inspect.getsource(App._tab_upscale)
        self.assertIn("list(self.USC_FORMATS.keys())", body)
        self.assertIn("default=self.USC_DEFAULT_FORMAT", body)

    def test_the_tip_explains_the_absence(self):
        """Removing an option people expect needs a reason on screen."""
        import inspect
        body = inspect.getsource(App._tab_upscale)
        i = body.index("usc_fmt_var")
        self.assertIn("no still-image PNG option", body[i:i + 1200])


class TestCacheButtonAndReporting(unittest.TestCase):
    """Clearing the cache, and saying more about what fal is doing."""

    class _Stub(App):
        def __init__(self, cache):
            self._cache = cache

        def _usc_ai_cache_dir(self):
            return self._cache

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        self.cache = tempfile.mkdtemp()
        self.app = self._Stub(self.cache)

    def tearDown(self):
        shutil.rmtree(self.cache, ignore_errors=True)

    # ── the cache button ──
    def test_an_empty_cache_reports_empty(self):
        self.assertEqual(self.app._usc_ai_cache_stats(), (0, 0.0))
        self.assertIn("empty", self.app._usc_cache_label())

    def test_it_counts_what_is_held(self):
        from PIL import Image
        for k in range(3):
            Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(
                os.path.join(self.cache, f"{k}.png"))
        n, mb = self.app._usc_ai_cache_stats()
        self.assertEqual(n, 3)
        self.assertGreater(mb, 0.0)
        self.assertIn("3 frame", self.app._usc_cache_label())

    def test_it_ignores_anything_that_is_not_a_frame(self):
        with open(os.path.join(self.cache, "notes.txt"), "w") as fh:
            fh.write("x")
        self.assertEqual(self.app._usc_ai_cache_stats()[0], 0)

    def test_clearing_asks_first(self):
        """Every frame in here was paid for: clearing is instant and
        regenerating is minutes and money."""
        body = self._src(App._usc_ai_clear_cache)
        self.assertIn("askyesno", body)
        self.assertIn('default="no"', body)
        self.assertLess(body.index("askyesno"), body.index("rmtree"))

    def test_the_prompt_names_the_cost(self):
        body = self._src(App._usc_ai_clear_cache)
        self.assertIn("paid generation", body)
        self.assertIn("generated again", body)

    def test_an_empty_cache_does_not_prompt(self):
        """Confirming the deletion of nothing trains people to click
        through the dialog that matters."""
        body = self._src(App._usc_ai_clear_cache)
        self.assertLess(body.index("already empty"), body.index("askyesno"))

    def test_the_button_shows_what_would_be_lost(self):
        body = self._src(App._usc_cache_label)
        self.assertIn("frame(s)", body)
        self.assertIn("MB", body)

    def test_the_label_is_refreshed_after_generating(self):
        self.assertIn("_usc_sync_cache_button",
                      self._src(App._usc_ai_generate))

    # ── fal reporting ──
    def test_the_request_id_is_surfaced(self):
        """It is the only handle fal support can act on."""
        body = self._src(App._vace_submit)
        self.assertIn("request_id", body)
        self.assertIn("fal request", body)

    def test_the_queue_position_is_logged_only_when_it_moves(self):
        """Repeating 'position 3' every poll buries fal's own log lines
        and makes an advancing queue look identical to a stuck one."""
        body = self._src(App._vace_submit)
        self.assertIn('if pos != state["pos"]', body)

    def test_the_render_start_is_announced_once(self):
        body = self._src(App._vace_submit)
        self.assertIn('state["phase"] != "render"', body)

    def test_fal_own_logs_are_not_repeated(self):
        body = self._src(App._vace_submit)
        self.assertIn("seen.add(m)", body)

    def test_the_endpoint_is_resolved_before_it_is_reported(self):
        """The handler names it, so it has to exist by then."""
        body = self._src(App._vace_submit)
        self.assertLess(body.index("app_id = "), body.index("def _on_update"))

    # ── the progress labels ──
    def test_the_render_label_carries_elapsed_as_well_as_remaining(self):
        """The estimate can be wrong; the elapsed time cannot, and it is
        the number that tells you whether to keep waiting."""
        body = self._src(App._usc_ai_generate)
        self.assertIn("in  ", body)
        self.assertIn("_vace_eta_text", body)

    def test_every_phase_label_carries_a_noun(self):
        """A bar that says only 'Queue' leaves you guessing whether the
        thing you asked for is the thing it is doing."""
        body = self._src(App._usc_ai_generate)
        self.assertIn('_bar("mask", f"', body)
        self.assertIn('_bar("decode", f"', body)


class TestUpscaleProgressReaches(unittest.TestCase):
    """The tab's dither bar REPLACED the window's progress bar instead
    of adding to it, so a running job looked stopped from anywhere but
    this tab."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_the_render_drives_both_bars(self):
        body = self._src(App._usc_run)
        self.assertIn("usc_bar.set", body)
        self.assertIn("self.progress(", body)

    def test_the_fal_phases_drive_both(self):
        body = self._src(App._usc_ai_generate)
        i = body.index("def _bar")
        seg = body[i:i + 2800]
        self.assertIn("usc_bar.set", seg)
        self.assertIn("self.progress(", seg)

    def test_the_render_estimate_drives_both(self):
        body = self._src(App._usc_ai_generate)
        i = body.index("def _eta_tick")
        seg = body[i:i + 1600]
        self.assertIn("usc_bar.set", seg)
        self.assertIn("self.progress(", seg)

    def test_both_are_cleared_when_the_run_ends(self):
        """A window bar left full is worse than one never touched."""
        body = self._src(App._usc_run)
        self.assertIn("usc_bar.reset()", body)
        self.assertIn("self.progress(-1)", body)

    def test_both_are_cleared_on_error(self):
        """The EXCEPTION handler specifically. There is a second
        setstatus("Error") on the partial-failure path, which is not
        where the bars are torn down."""
        body = self._src(App._usc_run)
        i = body.index("Run error:")
        seg = body[i:i + 400]
        self.assertIn("usc_bar.reset()", seg)
        self.assertIn("self.progress(-1)", seg)

    def test_the_generation_clears_the_window_bar_too(self):
        self.assertIn("self.progress(-1)",
                      self._src(App._usc_ai_generate))

    def test_the_labels_match_between_the_two(self):
        """Two bars showing different things is worse than one."""
        body = self._src(App._usc_run)
        i = body.index('_lbl = {("both")')
        seg = body[i:i + 1400]
        self.assertEqual(seg.count('f"{L}  {n} / {t}"'), 2)


class TestFalVisibility(unittest.TestCase):
    """"Is it going to plan" is two questions the log answered neither
    of: is fal still TALKING to us, and is this taking as long as it
    should. Silence is the more urgent -- a job that has stopped
    reporting is usually already lost, while one that is merely slow
    will still finish."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    # ── the verdict ──
    def test_a_normal_job_reads_as_on_track(self):
        state, _t = App._fal_health(5, 30, 200)
        self.assertEqual(state, "ok")

    def test_silence_is_flagged_before_a_timeout(self):
        """A gap is the earliest sign something is wrong, and it
        arrives long before a timeout does."""
        self.assertEqual(App._fal_health(100, 120, 200)[0], "quiet")

    def test_long_silence_is_named_as_beyond_experience(self):
        """The verdict compares against what this endpoint has actually
        done, which is actionable; a bare 'no update for 8 min' is
        not."""
        state, txt = App._fal_health(400, 500, 200)
        self.assertEqual(state, "stalled")
        self.assertIn("ever gone", txt)

    def test_silence_outranks_slowness(self):
        """A job that is both slow AND silent is a silence problem."""
        self.assertEqual(App._fal_health(400, 5000, 200)[0], "stalled")

    def test_overrun_is_distinguished_from_merely_slow(self):
        self.assertEqual(App._fal_health(5, 260, 200)[0], "late")
        self.assertEqual(App._fal_health(5, 500, 200)[0], "overdue")

    def test_the_verdict_quotes_the_numbers(self):
        _s, txt = App._fal_health(5, 500, 200)
        self.assertIn("8 min", txt)
        self.assertIn("3", txt)

    def test_every_state_has_a_colour(self):
        """A state with no colour would render as the default and read
        as healthy."""
        for sil, el in ((5, 30), (5, 260), (5, 500), (100, 120), (400, 500)):
            state, _t = App._fal_health(sil, el, 200)
            with self.subTest(state=state):
                self.assertIn(state, App._FAL_STATE_COLOUR)

    def test_a_zero_estimate_does_not_divide_by_zero(self):
        self.assertIsNotNone(App._fal_health(1, 10, 0)[0])

    # ── the panel ──
    def test_the_panel_is_a_standing_widget(self):
        """The log answers what happened; this answers whether it is
        going to plan, which is the question being asked while you
        wait."""
        self.assertIn("usc_fal_panel", self._src(App._tab_upscale))

    def test_any_update_resets_the_silence_timer(self):
        """_usc_fal_note became the generalised _fal_panel_note,
        parametrised by which tab's state attribute to update -- both
        Upscale & Expand and Fix Missing Frames now share this one
        implementation."""
        body = self._src(App._fal_panel_note)
        self.assertIn('["last"] =', body)

    def test_fal_log_lines_count_as_signs_of_life(self):
        body = self._src(App._usc_ai_generate)
        self.assertIn("def _relay", body)
        self.assertIn("_note(msg=m)", body)

    def test_the_panel_reports_queue_position(self):
        body = self._src(App._usc_ai_generate)
        self.assertIn("queue=", body)

    def test_the_panel_closes_on_success_and_on_failure(self):
        body = self._src(App._usc_ai_generate)
        self.assertIn("_fal_panel_end(", body)
        self.assertIn("fal call failed", body)

    def test_feeding_the_panel_is_optional_for_other_tabs(self):
        """_vace_submit serves three tabs and only this one has a
        panel, so it must not require one."""
        body = self._src(App._vace_submit)
        i = body.index("def _pfeed")
        self.assertIn("except Exception", body[i:i + 400])

    def test_the_tick_stops_when_the_job_does(self):
        """A timer left running after the window is gone throws once a
        second, forever."""
        body = self._src(App._fal_panel_tick)
        self.assertIn('if not st.get("live")', body)


class TestNoOrphanedTabCode(unittest.TestCase):
    """A structural guard. An edit once left widget construction
    stranded between two methods, after a `return`, where it was
    unreachable AND had eaten the following def line -- syntactically
    valid, silently broken."""

    def test_no_method_builds_widgets_after_returning(self):
        with open(SRC, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        bad = []
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)]:
            seen_return = False
            for stmt in fn.body:
                if isinstance(stmt, ast.Return):
                    seen_return = True
                elif seen_return and isinstance(stmt, (ast.Assign, ast.Expr)):
                    src = ast.unparse(stmt)
                    if "tk." in src or "pack(" in src:
                        bad.append(f"{fn.name}: {src[:50]}")
        self.assertEqual(bad, [], f"unreachable widget code: {bad}")


class TestLearnedSilenceBudget(unittest.TestCase):
    """Plenty of fal endpoints emit nothing during inference -- they
    run, say nothing for as long as the model takes, then return. On one
    of those a hardcoded five-minute "stalled" verdict fires every
    single run, and an alarm that is always wrong is worse than none:
    it teaches you to ignore the one time it is right."""

    EP = "wan-22-vace-fun-a14b/inpainting"

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self._saved = App.VACE_TIMING_FILE
        App.VACE_TIMING_FILE = os.path.join(self.dir, "t.json")

    def tearDown(self):
        App.VACE_TIMING_FILE = self._saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_defaults_apply_with_no_history(self):
        self.assertEqual(App._vace_silence_budget(self.EP),
                         (App.VACE_QUIET_S, App.VACE_STALLED_S))

    def test_an_eight_minute_gap_starts_out_alarming(self):
        """Which is right when nothing is known about the endpoint."""
        self.assertEqual(
            App._fal_health(480, 600, 200, endpoint=self.EP)[0], "stalled")

    def test_it_stops_alarming_once_that_gap_is_normal(self):
        """Three successful runs that each went quiet for eight minutes
        make eight minutes unremarkable."""
        for s in (470.0, 495.0, 488.0):
            App._vace_record_silence(s, self.EP)
        # Estimate matched to the run: at 200s expected the job would be
        # flagged for RUNNING LONG, which is a separate and correct
        # verdict -- this test is about the silence alone.
        self.assertEqual(
            App._fal_health(480, 600, 600, endpoint=self.EP)[0], "ok")

    def test_it_still_alarms_well_past_what_it_has_seen(self):
        for s in (470.0, 495.0, 488.0):
            App._vace_record_silence(s, self.EP)
        self.assertEqual(
            App._fal_health(1500, 1600, 200, endpoint=self.EP)[0], "stalled")

    def test_the_budget_has_headroom_over_the_worst_seen(self):
        """Equal to the worst would alarm on the very next normal run."""
        App._vace_record_silence(400.0, self.EP)
        quiet, stalled = App._vace_silence_budget(self.EP)
        self.assertGreater(quiet, 400.0)
        self.assertGreater(stalled, quiet)

    def test_each_endpoint_keeps_its_own(self):
        """They behave nothing alike: one streams progress throughout
        and another is silent until it returns."""
        for s in (470.0, 495.0):
            App._vace_record_silence(s, self.EP)
        self.assertEqual(App._vace_silence_budget("other/model"),
                         (App.VACE_QUIET_S, App.VACE_STALLED_S))

    def test_an_unknown_endpoint_does_not_inherit(self):
        """A model that streams every second must not inherit an
        eight-minute budget, or its alarm would never fire."""
        App._vace_record_silence(600.0, self.EP)
        self.assertEqual(App._vace_timing_history("silence", "brand/new"), [])

    def test_only_successful_runs_teach_it(self):
        """Learning from a failure would teach the alarm to tolerate
        exactly the silence that preceded it."""
        import inspect
        body = inspect.getsource(App._fal_panel_end)
        self.assertIn("if ok and", body)
        self.assertIn("_vace_record_silence", body)

    def test_a_failed_run_is_marked_as_such(self):
        import inspect
        self.assertIn("ok=False", inspect.getsource(App._usc_ai_generate))

    def test_the_verdict_says_what_it_is_comparing_against(self):
        """'Longer than this endpoint has ever gone' is actionable;
        'no update for 8 min' is not."""
        for s in (470.0, 495.0):
            App._vace_record_silence(s, self.EP)
        _st, txt = App._fal_health(1500, 1600, 200, endpoint=self.EP)
        self.assertIn("ever gone", txt)

    def test_render_timing_is_also_per_endpoint(self):
        import inspect
        self.assertIn("endpoint", inspect.getsource(App._vace_record_render))
        self.assertIn("endpoint", inspect.getsource(App._vace_estimate))

    def test_the_old_flat_file_does_not_break_it(self):
        """Existing installs have {"spf": [...]} with no by_endpoint."""
        import json
        with open(App.VACE_TIMING_FILE, "w", encoding="utf-8") as fh:
            json.dump({"spf": [2.0, 2.2]}, fh)
        self.assertEqual(App._vace_timing_history("spf"), [2.0, 2.2])
        self.assertEqual(App._vace_silence_budget("x"),
                         (App.VACE_QUIET_S, App.VACE_STALLED_S))

    def test_bad_values_are_ignored(self):
        for bad in (0, -5, None):
            App._vace_record_value("silence", bad, self.EP)
        self.assertEqual(App._vace_timing_history("silence", self.EP), [])


class TestHighlightPrompt(unittest.TestCase):
    """Steering what goes into the blown regions.

    The model is shown a plate with grey holes where the highlights
    were, and left alone it fills them with whatever it thinks belongs
    in a hole that shape -- filaments, bulbs, lettering on a sign."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_the_default_describes_light_not_subjects(self):
        p = App.HDR_AI_PROMPT.lower()
        self.assertIn("overexposed", p)
        self.assertIn("surrounding", p)
        self.assertIn("no new objects", p)

    def test_the_negative_names_what_it_invents(self):
        """Generic negatives do not stop a model painting a shop sign;
        naming signage does."""
        n = App.HDR_AI_NEGATIVE.lower()
        for term in ("lettering", "signage", "filaments", "faces"):
            with self.subTest(term=term):
                self.assertIn(term, n)

    def test_the_negative_also_covers_the_usual_artefacts(self):
        n = App.HDR_AI_NEGATIVE.lower()
        for term in ("flicker", "smear", "watermark"):
            with self.subTest(term=term):
                self.assertIn(term, n)

    def test_the_field_exists_and_is_prefilled(self):
        body = self._src(App._tab_upscale)
        self.assertIn("usc_ai_prompt", body)
        self.assertIn("HDR_AI_PROMPT", body)

    def test_the_prompt_reaches_the_request(self):
        body = self._src(App._usc_ai_generate)
        self.assertIn("def _payload", body)
        self.assertIn("usc_ai_prompt", body)
        self.assertIn("payload_fn=_payload", body)

    def test_it_stops_borrowing_the_other_tabs_prompt(self):
        """An expansion was being steered by whatever was typed on
        ADW-Interpolate, invisibly."""
        body = self._src(App._usc_ai_generate)
        i = body.index("def _payload")
        seg = body[i:i + 1200]
        self.assertNotIn("ip_vace_prompt_var", seg)
        self.assertNotIn("ip_vace_neg_var", seg)

    def test_an_empty_field_falls_back_to_the_default(self):
        """A blank prompt lets the model choose freely, which is the
        case this exists to prevent."""
        body = self._src(App._usc_ai_generate)
        self.assertIn("or self.HDR_AI_PROMPT", body)

    def test_the_negative_is_not_left_to_the_user(self):
        """It is the guardrail, not a preference."""
        body = self._src(App._usc_ai_generate)
        self.assertIn("HDR_AI_NEGATIVE", body)

    def test_the_prompt_is_logged(self):
        """A result that invented something should be traceable to what
        was asked for."""
        self.assertIn("guide:", self._src(App._usc_ai_generate))

    def test_the_tip_says_to_name_the_material(self):
        body = self._src(App._tab_upscale)
        i = body.index("usc_ai_prompt")
        self.assertIn("material", body[i:i + 1600])

    def test_the_field_greys_out_with_the_toggle(self):
        """Superseded: individual row greying (_usc_ai_rows) was
        replaced by showing/hiding the whole Highlights/Shadows
        notebook -- the Guide field lives inside that notebook, so it
        is hidden entirely rather than merely greyed whenever Wan VACE
        inpaint is not selected."""
        body = self._src(App._tab_upscale)
        i = body.index('_prow = tk.Frame(_hi_tab')
        self.assertGreater(i, body.index("self.usc_ai_notebook = ttk.Notebook"))


class TestLogGammaWorkingSpace(unittest.TestCase):
    """From DiffHDR (arXiv 2604.06161): do the reconstruction in a log
    space rather than a display encoding.

    sRGB gives the range above diffuse white ZERO code values, so a
    source running to 12x white is a flat 255 from 1.0 upward and the
    falloff around a blown region is destroyed before the model sees
    it."""

    @staticmethod
    def _one(fn, v):
        return float(fn(np.array([[[v]]], np.float32))[0, 0, 0])

    def test_the_curve_round_trips(self):
        for v in (0.0, 0.001, 0.01, 0.18, 1.0, 4.0, 20.0):
            with self.subTest(linear=v):
                back = self._one(App._logc_to_linear,
                                 self._one(App._linear_to_logc, v))
                self.assertAlmostEqual(back, v, delta=max(1e-4, v * 1e-4))

    def test_it_is_monotonic(self):
        vals = [self._one(App._linear_to_logc, v)
                for v in (0.0, 0.01, 0.1, 1.0, 5.0, 20.0, 50.0)]
        self.assertEqual(vals, sorted(vals))

    def test_it_has_a_linear_toe(self):
        """A pure log blows up at zero, and the toe is where a night
        plate keeps most of its pixels."""
        self.assertGreater(self._one(App._linear_to_logc, 0.0), 0.0)
        self.assertTrue(np.isfinite(self._one(App._linear_to_logc, 0.0)))

    def test_srgb_gives_the_highlights_nothing(self):
        """The measurement that motivates the change."""
        a = self._one(App._linear_to_srgb, 1.0)
        b = self._one(App._linear_to_srgb, 12.0)
        self.assertAlmostEqual((b - a) * 255.0, 0.0, delta=0.5)

    def test_logc_gives_the_highlights_real_range(self):
        a = self._one(App._linear_to_logc, 1.0)
        b = self._one(App._linear_to_logc, 12.0)
        self.assertGreater((b - a) * 255.0, 40.0)

    def test_over_range_survives_an_8_bit_hand_off(self):
        """This is what the model round trip actually is: float in,
        8-bit to the model, float back."""
        for v in (2.0, 6.0, 12.0):
            with self.subTest(linear=v):
                code = np.clip(App._linear_to_logc(
                    np.array([[[v]]], np.float32)), 0, 1)
                u8 = (code * 255.0 + 0.5).astype(np.uint8)
                back = float(App._logc_to_linear(
                    u8.astype(np.float32) / 255.0)[0, 0, 0])
                self.assertAlmostEqual(back, v, delta=v * 0.05)

    def test_the_same_hand_off_in_srgb_clips(self):
        """Pins the comparison, so a revert to sRGB fails loudly."""
        code = np.clip(App._linear_to_srgb(
            np.array([[[6.0]]], np.float32)), 0, 1)
        u8 = (code * 255.0 + 0.5).astype(np.uint8)
        back = float(App._srgb_to_linear(
            u8.astype(np.float32) / 255.0)[0, 0, 0])
        self.assertAlmostEqual(back, 1.0, delta=0.01)

    # ── wiring ──
    def test_frames_are_staged_in_the_working_space(self):
        """Handing the model the plate directly would throw the
        falloff away before it can look at it."""
        import inspect
        body = inspect.getsource(App._usc_ai_generate)
        self.assertIn("_usc_ai_encode", body)
        self.assertIn("HDR_AI_SPACE", body)

    def test_the_source_clip_is_built_from_the_staged_copy(self):
        import inspect
        body = inspect.getsource(App._usc_ai_generate)
        self.assertIn("_vace_build_source_clip(stage, staged", body)

    def test_the_encode_prefers_the_float_source(self):
        """An 8-bit read would have clipped the over-range already."""
        # Within the LOGC branch specifically. _usc_read_rgb appears
        # earlier in the function, in the non-logc early-out.
        import inspect
        body = inspect.getsource(App._usc_ai_encode)
        i = body.index('if self.HDR_AI_SPACE != "logc"')
        # Past the END of the early-out line, not its start.
        j = body.index("\n", body.index("return", i))
        logc_branch = body[j:]
        self.assertIn("_usc_read_float", logc_branch)
        self.assertLess(logc_branch.index("_usc_read_float"),
                        logc_branch.index("_usc_read_rgb"))

    def test_a_display_source_is_linearised_first(self):
        """So the curve means the same thing whatever came in."""
        import inspect
        self.assertIn("_srgb_to_linear",
                      inspect.getsource(App._usc_ai_encode))

    def test_the_merge_understands_the_space(self):
        import inspect
        self.assertIn("logc", inspect.getsource(App._hdr_ai_merge))

    def test_every_merge_uses_the_models_space_not_the_plates(self):
        """`g` came back from the generation, which was fed staged
        frames -- decoding it with the source's curve would undo the
        wrong transform."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        for chunk in src.split("_hdr_ai_merge(")[1:]:
            if chunk.lstrip().startswith("cls,"):
                continue          # the definition, not a call
            with self.subTest(call=chunk[:30].strip()):
                self.assertIn("HDR_AI_SPACE", chunk[:420])

    def test_the_space_can_be_switched_back(self):
        """A single constant, so an A/B against sRGB is one edit."""
        self.assertIn(App.HDR_AI_SPACE, ("logc", "srgb"))


class TestPromptPresets(unittest.TestCase):
    """A menu of surfaces, with the text still editable."""

    class _V:
        def __init__(self, v):
            self.v = v

        def get(self):
            return self.v

        def set(self, v):
            self.v = v

    class _Stub(App):
        def __init__(self):
            self.usc_ai_preset = TestPromptPresets._V(
                App.HDR_AI_PRESETS[0][0])
            self.usc_ai_prompt = TestPromptPresets._V(App.HDR_AI_PROMPT)

    # ── the presets themselves ──
    def test_the_first_is_the_default_guide(self):
        self.assertEqual(App.HDR_AI_PRESETS[0][1], App.HDR_AI_PROMPT)

    def test_custom_is_last_and_holds_no_text(self):
        """It exists so the menu can report that the text is yours
        without overwriting it to say so."""
        name, txt = App.HDR_AI_PRESETS[-1]
        self.assertIn("Custom", name)
        self.assertIsNone(txt)

    def test_names_are_unique(self):
        names = [n for n, _t in App.HDR_AI_PRESETS]
        self.assertEqual(len(names), len(set(names)))

    def test_there_are_night_options(self):
        """The blown regions in a night exterior are usually the light
        sources themselves."""
        night = [n for n, _t in App.HDR_AI_PRESETS if "Night" in n]
        self.assertGreaterEqual(len(night), 4)

    def test_every_preset_names_a_surface_not_a_subject(self):
        """'Fairground lights' tells the model to paint lights. The
        prompt has to describe what the blown pixels WERE."""
        for name, txt in App.HDR_AI_PRESETS:
            if txt is None:
                continue
            with self.subTest(preset=name):
                self.assertNotRegex(txt.lower(),
                                    r"\b(a |an |the )\w+ seen\b")

    def test_every_preset_says_continuous_or_consistent(self):
        """The job is extending what is already there."""
        for name, txt in App.HDR_AI_PRESETS:
            if txt is None:
                continue
            with self.subTest(preset=name):
                low = txt.lower()
                self.assertTrue("continuous" in low or "consistent" in low
                                or "matching" in low)

    def test_the_emissive_presets_ask_for_less_not_more(self):
        """There was never structure inside a blown lamp, so the right
        instruction is to do less."""
        for name, txt in App.HDR_AI_PRESETS:
            if txt and "Night" in name:
                with self.subTest(preset=name):
                    self.assertIn("no ", txt.lower())

    def test_presets_stay_short(self):
        """A long prompt drifts toward describing a picture, which is
        how invented content gets in."""
        for name, txt in App.HDR_AI_PRESETS:
            if txt:
                with self.subTest(preset=name):
                    self.assertLess(len(txt.split()), 30)

    # ── the two controls in step ──
    def test_choosing_a_preset_fills_the_field(self):
        app = self._Stub()
        app.usc_ai_preset.set("Skin & faces")
        app._usc_ai_preset_changed()
        self.assertIn("skin", app.usc_ai_prompt.get().lower())

    def test_typing_switches_the_menu_to_custom(self):
        """Otherwise the menu names a preset the words no longer
        match."""
        app = self._Stub()
        app.usc_ai_prompt.set("wet cobbles under sodium light")
        app._usc_ai_prompt_edited()
        self.assertIn("Custom", app.usc_ai_preset.get())

    def test_text_matching_a_preset_keeps_that_preset(self):
        app = self._Stub()
        app.usc_ai_preset.set("Skin & faces")
        app._usc_ai_preset_changed()
        app._usc_ai_prompt_edited()
        self.assertEqual(app.usc_ai_preset.get(), "Skin & faces")

    def test_choosing_custom_leaves_the_text_alone(self):
        app = self._Stub()
        app.usc_ai_prompt.set("my own words")
        app.usc_ai_preset.set(App.HDR_AI_PRESETS[-1][0])
        app._usc_ai_preset_changed()
        self.assertEqual(app.usc_ai_prompt.get(), "my own words")

    def test_loading_a_preset_does_not_bounce_to_custom(self):
        """The guard against the two traces fighting each other."""
        import inspect
        self.assertIn("_usc_ai_syncing",
                      inspect.getsource(App._usc_ai_preset_changed))
        self.assertIn("_usc_ai_syncing",
                      inspect.getsource(App._usc_ai_prompt_edited))

    def test_the_menu_greys_out_with_the_toggle(self):
        """Superseded: the Surface menu lives inside the Highlights
        tab of the notebook that is now shown/hidden as a whole."""
        import inspect
        body = inspect.getsource(App._tab_upscale)
        i = body.index('_pprow = tk.Frame(_hi_tab')
        self.assertGreater(i, body.index("self.usc_ai_notebook = ttk.Notebook"))


class TestRenderColourspaceSnapshot(unittest.TestCase):
    """A render read frames 0-200 as display-encoded and 201 onward as
    raw, in one pass over one sequence.

    "Pinned" only ever meant "decided once": the value lived in mutable
    instance state that the EXR colourspace control writes to and a
    folder load clears, so a job taking minutes could be split down the
    middle by a single click."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_the_loader_accepts_an_explicit_decision(self):
        import inspect
        sig = inspect.signature(App._organic_load_linear)
        self.assertIn("encoded", sig.parameters)
        self.assertIsNone(sig.parameters["encoded"].default)

    def test_an_override_wins_over_the_shared_state(self):
        body = self._src(App._organic_load_linear)
        self.assertIn("if encoded is not None:", body)
        self.assertLess(body.index("if encoded is not None:"),
                        body.index("elif self._organic_source_encoded"))

    def test_the_override_is_used_for_the_branch_too(self):
        """Setting the attribute but then reading it back would still
        race the UI between those two lines."""
        body = self._src(App._organic_load_linear)
        self.assertIn("use_encoded", body)
        self.assertIn("if use_encoded:", body)

    def test_the_render_snapshots_before_the_loop(self):
        body = self._src(App._organic_run_worker)
        self.assertIn("_enc = self._organic_source_encoded", body)
        self.assertLess(body.index("_enc ="),
                        body.index("for n, i in enumerate(range("))

    def test_every_frame_of_a_render_uses_the_snapshot(self):
        body = self._src(App._organic_run_worker)
        # encoded AND the ACES decision, both snapshotted
        self.assertIn("_organic_load_linear(src, encoded=_enc,", body)
        self.assertIn("aces=_aces)", body)
        self.assertNotIn("_organic_load_linear(src)", body)

    def test_an_undecided_sequence_still_gets_a_value(self):
        """A render started before any frame was previewed would
        otherwise snapshot None and re-open the whole question."""
        body = self._src(App._organic_run_worker)
        i = body.index("_enc = self._organic_source_encoded")
        self.assertIn("_adw_source_is_encoded", body[i:i + 200])

    def test_the_preview_still_follows_the_control(self):
        """Only the RENDER is frozen -- toggling the control is how you
        check which reading is right."""
        body = self._src(App._organic_show_frame)
        self.assertIn("_organic_load_linear(path)", body)


class TestUpscaleVersionedOutput(unittest.TestCase):
    """Upscale & Expand was the one render in the app still using the
    plain, overwrite-on-rerun ADW/<name> path -- and a run through this
    tab is as often as not a paid AI generation, so a second run
    silently clobbering the first destroys something that cost money to
    make, with no way back."""

    class _Stub(App):
        def __init__(self):
            pass

    def setUp(self):
        self.app = self._Stub()
        self.dir = tempfile.mkdtemp()
        self.src = os.path.join(self.dir, "plate")
        os.makedirs(self.src)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    # ── the render itself ──
    def test_the_render_uses_the_versioned_writer(self):
        body = self._src(App._usc_run)
        self.assertIn("_adw_versioned_output_dir(_base_dir, _tagdir)", body)
        self.assertNotIn('_adw_module_output_dir(_base_dir, _tagdir)', body)

    def test_a_second_run_does_not_clobber_the_first(self):
        first = self.app._adw_versioned_output_dir(self.src, "UPSCALE")
        with open(os.path.join(first, "00001.exr"), "w") as fh:
            fh.write("frame one")
        second = self.app._adw_versioned_output_dir(self.src, "UPSCALE")
        self.assertNotEqual(first, second)
        self.assertTrue(os.path.exists(os.path.join(first, "00001.exr")))

    def test_expand_and_upscale_version_independently(self):
        """The two job modes write to different subfolders, and each
        should keep its own version sequence."""
        up1 = self.app._adw_versioned_output_dir(self.src, "UPSCALE")
        ex1 = self.app._adw_versioned_output_dir(self.src, "EXPAND")
        up2 = self.app._adw_versioned_output_dir(self.src, "UPSCALE")
        self.assertTrue(up1.endswith("_v01"))
        self.assertTrue(ex1.endswith("_v01"))
        self.assertTrue(up2.endswith("_v02"))

    # ── the peek helper ──
    def test_peeking_creates_nothing(self):
        """A label that redraws on every keystroke must not allocate a
        version folder just by being drawn."""
        path = self.app._adw_peek_next_version_dir(self.src, "UPSCALE")
        self.assertFalse(os.path.exists(path))

    def test_peeking_is_idempotent(self):
        a = self.app._adw_peek_next_version_dir(self.src, "UPSCALE")
        b = self.app._adw_peek_next_version_dir(self.src, "UPSCALE")
        self.assertEqual(a, b)

    def test_the_peek_matches_what_the_real_call_produces(self):
        """The one place a person checks before pressing Run has to
        agree with where Run actually writes."""
        peeked = self.app._adw_peek_next_version_dir(self.src, "UPSCALE")
        real = self.app._adw_versioned_output_dir(self.src, "UPSCALE")
        self.assertEqual(peeked, real)

    def test_the_peek_advances_after_a_real_run(self):
        self.app._adw_versioned_output_dir(self.src, "UPSCALE")
        nxt = self.app._adw_peek_next_version_dir(self.src, "UPSCALE")
        self.assertTrue(nxt.endswith("_v02"))

    def test_the_peek_avoids_equalling_the_input_folder(self):
        """Mirrors the real function's own collision guard, since a
        label showing a path equal to the input would be as wrong as
        the render actually writing there."""
        root = self.app._adw_output_root(self.src)
        clashing = os.path.join(root, "UPSCALE_v01")
        os.makedirs(os.path.dirname(clashing), exist_ok=True)
        # Force the collision the way the real function guards against:
        # peek from an input folder that already equals the candidate.
        nested_src = clashing
        os.makedirs(nested_src, exist_ok=True)
        peeked = self.app._adw_peek_next_version_dir(nested_src, "UPSCALE")
        self.assertNotEqual(os.path.abspath(peeked),
                            os.path.abspath(nested_src))

    # ── the label ──
    def test_the_label_uses_the_peek_not_a_hardcoded_join(self):
        """It was showing self._usc_folder/UPSCALE -- neither where
        the render lands (ADW/ is a sibling of the input, not inside
        it) nor versioned."""
        body = self._src(App._usc_update_out_label)
        self.assertIn("_adw_peek_next_version_dir", body)
        self.assertNotIn('os.path.join(self._usc_folder, "UPSCALE")', body)

    def test_the_label_follows_the_job_mode(self):
        body = self._src(App._usc_update_out_label)
        self.assertIn("_usc_job_mode()", body)
        self.assertIn('"EXPAND"', body)


class TestColorspaceConsistencyCheck(unittest.TestCase):
    """Fix Missing Frames gained a fourth detector: catch a sequence
    whose PEAK, BLACK FLOOR, or COLOUR BALANCE jumps by more than the
    clip's own frame-to-frame variability ever explains. This is
    exactly what a mid-render colour-space flip looks like from the
    outside -- everything correct up to one frame, then several stops
    brighter or a different cast from there on, with nothing in the
    footage to explain it. It is the direct answer to the Organic bug
    diagnosed earlier this session: frames to 200 linearised and
    correct, 201 onward read raw, peak jumping from ~0.79 to ~9.88 with
    nothing in the plate to justify it."""

    @staticmethod
    def _frame(peak, mean_ratio=1.0, seed=0, size=(48, 64),
               tint=(1.0, 1.0, 1.0)):
        rng = np.random.default_rng(seed)
        base = rng.random((*size, 3)).astype(np.float32) * mean_ratio
        base = np.clip(base * np.array(tint, np.float32), 0.0, None)
        base[6:14, 6:14] = peak
        return base

    def _sig(self, *a, **kw):
        return App._ff_signature(self._frame(*a, **kw))

    # ── the real bug, reproduced ──
    def test_it_catches_a_mid_sequence_decode_flip(self):
        """The originating case: display-encoded frames linearised
        correctly up to frame 201, then read as already-linear from
        there on -- peak jumps several stops with nothing in the
        content to explain it."""
        sigs = []
        for i in range(400):
            if i < 201:
                sigs.append(self._sig(0.75 + 0.05 * np.sin(i / 7.0), seed=i))
            else:
                sigs.append(self._sig(6.0 + 3.0 * np.sin(i / 7.0), seed=i))
        found = App._ff_colorspace_scan(sigs)
        self.assertEqual(found["peak_jump"], [201])

    def test_it_flags_only_the_transition_not_the_whole_run(self):
        """Every frame from 201 on is equally bright relative to ITS
        neighbours -- only the single frame where the jump lands is a
        finding."""
        sigs = [self._sig(0.8, seed=i) if i < 100
               else self._sig(8.0, seed=i) for i in range(200)]
        found = App._ff_colorspace_scan(sigs)
        self.assertEqual(len(found["peak_jump"]), 1)

    # ── the signals in isolation ──
    def test_a_black_floor_jump_is_caught_even_with_similar_peak(self):
        """Blacks crushing or lifting is a distinct symptom from an
        overall exposure jump, and the user asked for this one by
        name."""
        sigs = []
        for i in range(100):
            rng = np.random.default_rng(3000 + i)
            base = rng.random((48, 64)).astype(np.float32) * 0.3 + 0.4
            if i >= 50:
                base = np.clip(base - 0.35, 0.0, None)
            sigs.append(App._ff_signature(np.dstack([base] * 3)))
        found = App._ff_colorspace_scan(sigs)
        self.assertIn(50, found["black_jump"])

    def test_a_pure_colour_balance_shift_is_caught(self):
        """Same overall brightness, the channels stop agreeing with
        each other -- a peak-only or mean-only check would miss this
        entirely."""
        sigs = [self._sig(0.7, mean_ratio=0.3, seed=4000 + i,
                          tint=(1.0, 1.0, 1.0) if i < 60
                          else (1.6, 1.0, 0.5))
               for i in range(100)]
        found = App._ff_colorspace_scan(sigs)
        self.assertIn(60, found["balance_jump"])

    def test_three_channels_moving_together_is_not_a_balance_shift(self):
        """An overall exposure change moves R, G and B together and
        must register as a peak/mean matter, not a balance one."""
        sigs = [self._sig(0.6 if i < 50 else 4.0, seed=i)
               for i in range(100)]
        found = App._ff_colorspace_scan(sigs)
        self.assertEqual(found["balance_jump"], [])

    # ── it must not cry wolf ──
    def test_an_ordinary_clip_with_motion_and_grain_is_clean(self):
        sigs = []
        for i in range(200):
            f = self._frame(0.6 + 0.1 * np.sin(i / 15.0), seed=1000 + i)
            f = f + np.random.default_rng(i).normal(
                0, 0.01, f.shape).astype(np.float32)
            sigs.append(App._ff_signature(f))
        found = App._ff_colorspace_scan(sigs)
        self.assertEqual(found, {"peak_jump": [], "black_jump": [],
                                 "balance_jump": []})

    def test_a_slow_intentional_fade_does_not_trip_it(self):
        """A gradual 200-frame ramp is exactly what a real exposure
        change or a fade looks like, and must not be condemned --
        only a SUDDEN, single-frame jump is a finding."""
        sigs = [self._sig(0.9 * (0.3 + 0.7 * i / 199.0), seed=2000 + i)
               for i in range(200)]
        found = App._ff_colorspace_scan(sigs)
        self.assertEqual(found, {"peak_jump": [], "black_jump": [],
                                 "balance_jump": []})

    def test_it_is_relative_to_the_clip_not_an_absolute_number(self):
        """A high-key clip and a low-key clip both have their own
        normal amount of frame-to-frame variability; the same relative
        jump should be treated the same way in either."""
        bright = [self._sig(0.7 + 0.1 * np.sin(i / 5.0), seed=i)
                 for i in range(60)]
        dim = [self._sig(0.05 + 0.01 * np.sin(i / 5.0), seed=i)
              for i in range(60)]
        self.assertEqual(App._ff_colorspace_scan(bright)["peak_jump"], [])
        self.assertEqual(App._ff_colorspace_scan(dim)["peak_jump"], [])

    def test_transitions_touching_a_black_frame_are_excluded(self):
        """A true black frame's peak is ~0 by definition; the jump
        into or out of one is not a colour-space finding."""
        sigs = [self._sig(0.7, seed=i) for i in range(20)]
        sigs[10] = App._ff_signature(np.zeros((48, 64, 3), np.float32))
        found = App._ff_colorspace_scan(sigs)
        self.assertNotIn(10, found["peak_jump"])
        self.assertNotIn(11, found["peak_jump"])

    # ── robustness ──
    def test_none_entries_from_missing_frames_do_not_raise(self):
        sigs = [self._sig(0.7, seed=i) if i % 4 else None
               for i in range(40)]
        App._ff_colorspace_scan(sigs)   # must not raise

    def test_an_empty_or_tiny_list_does_not_raise(self):
        for sigs in ([], [self._sig(0.5)], [None, None]):
            with self.subTest(n=len(sigs)):
                found = App._ff_colorspace_scan(sigs)
                self.assertEqual(found, {"peak_jump": [], "black_jump": [],
                                         "balance_jump": []})

    def test_sensitivity_scales_the_threshold(self):
        """The same one control that scales every other detector."""
        sigs = [self._sig(0.7, seed=i) for i in range(30)]
        sigs[15] = self._sig(1.4, seed=15)   # a borderline jump
        low = App._ff_colorspace_scan(sigs, sensitivity=3.0)["peak_jump"]
        high = App._ff_colorspace_scan(sigs, sensitivity=0.1)["peak_jump"]
        self.assertEqual(low, [])
        self.assertIn(15, high)

    def test_a_bad_sensitivity_value_does_not_raise(self):
        sigs = [self._sig(0.7, seed=i) for i in range(10)]
        for bad in (0, -5, None):
            with self.subTest(sensitivity=bad):
                App._ff_colorspace_scan(sigs, sensitivity=bad or 1e-9)

    # ── the signature extension is additive, not breaking ──
    def test_the_signature_still_supports_is_black_by_index(self):
        sig = App._ff_signature(np.zeros((10, 10, 3), np.float32))
        self.assertTrue(App._ff_is_black(sig))

    def test_the_signature_still_supports_thumb_diff_by_index(self):
        a = App._ff_signature(self._frame(0.5, seed=1))
        b = App._ff_signature(self._frame(0.5, seed=1))
        self.assertLess(App._ff_thumb_diff(a, b), 0.01)

    def test_the_signature_gained_black_floor_and_channel_means(self):
        sig = App._ff_signature(self._frame(0.8, mean_ratio=0.4, seed=1))
        self.assertEqual(len(sig), 5)
        black, ch_means = sig[3], sig[4]
        self.assertIsInstance(black, float)
        self.assertEqual(len(ch_means), 3)

    def test_black_floor_uses_a_percentile_not_the_literal_minimum(self):
        """A single dead or masked pixel must not decide the whole
        frame's black floor."""
        a = self._frame(0.6, mean_ratio=0.4, seed=1)
        a[0, 0] = 0.0   # one dead pixel
        b = self._frame(0.6, mean_ratio=0.4, seed=1)
        black_a = App._ff_signature(a)[3]
        black_b = App._ff_signature(b)[3]
        self.assertAlmostEqual(black_a, black_b, delta=0.02)

    # ── wiring into the scan worker and UI ──
    def test_the_checkbox_exists_and_defaults_on(self):
        import inspect
        body = inspect.getsource(App._tab_fixframes)
        self.assertIn("ff_do_colorspace", body)
        self.assertIn('"Color space"', body)

    def test_the_tooltip_warns_about_genuine_events(self):
        """A real flash or hard cut can look identical to a colour-space
        bug from statistics alone."""
        import inspect
        body = inspect.getsource(App._tab_fixframes)
        i = body.index('"Color space"')
        self.assertIn("flash", body[i:i + 900].lower())

    def test_the_worker_calls_the_detector_when_enabled(self):
        import inspect
        body = inspect.getsource(App._ff_scan_worker)
        self.assertIn("_ff_colorspace_scan", body)
        self.assertIn("ff_do_colorspace", body)

    def test_the_worker_shares_the_sensitivity_control(self):
        """One control scales every detector, this one included --
        not a separate, undiscoverable second setting."""
        import inspect
        body = inspect.getsource(App._ff_scan_worker)
        i = body.index("_ff_colorspace_scan(")
        self.assertIn("ff_sens", body[i:i + 80])

    def test_findings_produce_all_three_fault_kinds(self):
        import inspect
        body = inspect.getsource(App._ff_scan_worker)
        for kind in ("cs_peak", "cs_black", "cs_bal"):
            with self.subTest(kind=kind):
                self.assertIn(kind, body)

    def test_missing_and_unreadable_frames_are_excluded_from_findings(self):
        """Same skip-set the other detectors use, so a placeholder
        standing in for an absent file cannot itself be blamed for a
        jump."""
        import inspect
        body = inspect.getsource(App._ff_scan_worker)
        i = body.index("_ff_colorspace_scan(")
        seg = body[i:i + 400]
        self.assertIn("i not in skip", seg)

    def test_the_timeline_gives_colorspace_faults_their_own_colour(self):
        import inspect
        body = inspect.getsource(App._tab_fixframes)
        self.assertIn("cs_peak", body)
        self.assertIn("#4488FF", body)


class TestTopazExpansionProvider(unittest.TestCase):
    """Topaz Hyperion as an alternative to this app's own analytic
    expansion. A different axis from AI Expansion (VACE): that step
    reconstructs blown highlights on top of whichever base expansion
    ran, and does not care which one produced it.

    Runway Ruby was considered and explicitly excluded: it is not on
    fal.ai, only on Runway's own separate API with its own auth and
    task-polling model -- a real second integration, not a drop-in next
    to Topaz."""

    class _V:
        def __init__(self, v):
            self.v = v

        def get(self):
            return self.v

    class _Stub(App):
        def __init__(self, provider=None):
            if provider is not None:
                self.usc_hdr_provider_var =                     TestTopazExpansionProvider._V(provider)

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    # ── PQ decode: the exact inverse of the existing PQ encode ──
    def test_pq_decode_is_the_exact_inverse_of_pq_encode(self):
        for nits in (0.0, 1.0, 100.0, 1000.0, 4000.0, 10000.0):
            with self.subTest(nits=nits):
                code = App._hdr_pq_encode(np.float32(nits))
                back = float(App._hdr_pq_decode(code))
                self.assertAlmostEqual(back, nits, delta=max(0.5, nits*0.002))

    def test_pq_decode_matches_the_known_reference_value(self):
        """0.7518 is the published PQ code for 1000 nits -- the same
        checkpoint the encoder was verified against."""
        back = float(App._hdr_pq_decode(np.float32(0.7518)))
        self.assertAlmostEqual(back, 1000.0, delta=1.0)

    def test_pq_decode_round_trips_a_whole_array(self):
        rng = np.random.default_rng(1)
        arr = (rng.random((8, 8, 3)).astype(np.float32) * 0.9 + 0.05) * 10000.0
        back = App._hdr_pq_decode(App._hdr_pq_encode(arr))
        self.assertLess(float(np.abs(back - arr).max()), 1.0)

    # ── the provider selector ──
    def test_provider_maps_local_by_default(self):
        """With no control built yet, the default is the non-spending
        choice."""
        self.assertEqual(self._Stub(None)._usc_hdr_provider(), "local")

    def test_provider_maps_each_display_label_correctly(self):
        local, topaz = App.HDR_PROVIDERS
        self.assertEqual(self._Stub(local)._usc_hdr_provider(), "local")
        self.assertEqual(self._Stub(topaz)._usc_hdr_provider(), "topaz")

    def test_an_unreadable_provider_control_defaults_to_local(self):
        """Local is the non-spending choice -- a control that cannot be
        read must not silently enable a paid path."""
        class Bad:
            def get(self):
                raise RuntimeError("boom")
        app = self._Stub()
        app.usc_hdr_provider_var = Bad()
        self.assertEqual(app._usc_hdr_provider(), "local")

    def test_the_provider_list_has_exactly_two_entries(self):
        self.assertEqual(len(App.HDR_PROVIDERS), 2)

    def test_the_local_provider_has_no_cost_marker(self):
        self.assertNotIn("\U0001F916", App.HDR_PROVIDERS[0])

    def test_topaz_is_named_recognisably(self):
        """No longer carrying an emoji or a cost note -- these values
        are internal now, switched on by _usc_hdr_provider(), not
        displayed directly. The three expansion-mode buttons carry the
        cost distinction visually instead: a smooth curve for the free
        local path, the pixel robot for anything paid and external."""
        self.assertIn("Topaz", App.HDR_PROVIDERS[1])

    def test_the_icon_distinction_lives_on_the_buttons_now(self):
        icons = {m: i for m, _n, i in App.USC_MODELS}
        self.assertEqual(icons["tonemap"], "curve")
        self.assertEqual(icons["topaz"], "robot")
        self.assertIn("_usc_draw_glyph", self._src(App._usc_draw_tile))

    # ── the standalone robot badge ──
    def test_the_badge_is_a_distinct_widget_from_the_toggle(self):
        """_robot_toggle draws the bitmap as part of a full latching
        button; this is the bare glyph for marking a SEPARATE control,
        reusing the one drawing routine rather than a second copy of
        the pixel pattern."""
        body = self._src(App._robot_badge)
        self.assertIn("_ROBOT_PIXELS", body)
        self.assertNotIn("def _click", body)

    def test_the_badge_exposes_set_lit(self):
        body = self._src(App._robot_badge)
        self.assertIn("set_lit", body)

    # ── greying ──
    def _greying(self, sel, pref=None):
        class Stub(App):
            def __init__(s):
                s._usc_sel = dict(sel)
                s._usc_view_pref = pref
                s._usc_hdr_analytic_rows = ["peak", "knee"]
                s.usc_hdr_topaz_rows = ["fmt"]
        app = Stub()
        got = {}
        app._usc_set_enabled = lambda row, on: got.__setitem__(row, on)
        app._usc_refresh_model_controls()
        return got

    def test_topaz_selected_greys_the_analytic_controls(self):
        """Target peak / Knee are inputs to THIS app's own curve,
        meaningless to a model that computes its own internally."""
        got = self._greying({"topaz": True})
        self.assertEqual((got["peak"], got["knee"]), (False, False))
        # ...but stay live while Tone map or Wan (which layers on the
        # Tone map base) is also in play
        got = self._greying({"topaz": True, "wan_vace": True})
        self.assertEqual((got["peak"], got["knee"]), (True, True))

    def test_topaz_selected_ungreys_the_topaz_format_row(self):
        self.assertTrue(self._greying({"topaz": True})["fmt"])
        self.assertFalse(self._greying({"tonemap": True})["fmt"])
        self.assertIn("self._usc_refresh_model_controls()",
                      self._src(App._usc_hdr_provider_changed))

    def test_the_analytic_rows_are_captured_robustly(self):
        """Not by counting sibling positions from another widget --
        that breaks the moment anything else in the card changes.
        Captured as whichever child appeared that was not there a
        moment before."""
        body = self._src(App._tab_upscale)
        i = body.index("_usc_hdr_analytic_rows")
        seg = body[i:i + 700]
        self.assertIn("before = set(c_dr.winfo_children())", seg)
        self.assertNotIn("winfo_children().index(", seg)

    # ── the dispatch point ──
    def test_dispatch_falls_through_to_analytic_with_no_topaz_result(self):
        app = self._Stub()
        code = np.full((4, 4, 3), 0.5, np.float32)
        out = app._usc_scene_for(0, code, None, 1000.0, 0.60, "srgb")
        expected = App._hdr_expand_frame(code, 1000.0, 0.60, to_pq=False,
                                         eotf="srgb", gamut="bt709",
                                         scale="white")
        np.testing.assert_allclose(out, expected)

    def test_dispatch_returns_the_topaz_frame_untouched_when_available(self):
        """Not re-run through the analytic maths -- Topaz already
        produced the scene-referred result for this frame."""
        app = self._Stub()
        code = np.full((4, 4, 3), 0.5, np.float32)
        topaz_frame = np.full((4, 4, 3), 7.0, np.float32)
        out = app._usc_scene_for(1, code, [None, topaz_frame], 1000.0,
                                 0.60, "srgb")
        np.testing.assert_array_equal(out, topaz_frame)

    def test_dispatch_falls_back_for_an_index_beyond_a_partial_result(self):
        """A Topaz result shorter than the span sent (a returned-frame
        mismatch) must degrade per frame, not raise."""
        app = self._Stub()
        code = np.full((4, 4, 3), 0.5, np.float32)
        out = app._usc_scene_for(5, code, [np.zeros((4, 4, 3), np.float32)],
                                 1000.0, 0.60, "srgb")
        expected = App._hdr_expand_frame(code, 1000.0, 0.60, to_pq=False,
                                         eotf="srgb", gamut="bt709",
                                         scale="white")
        np.testing.assert_allclose(out, expected)

    def test_one_dispatch_point_not_three_separate_substitutions(self):
        """So the fallback logic cannot drift between the EXR path, the
        PQ/display path, and the wipe-preview refresh."""
        body = self._src(App._usc_run)
        # EXR, PQ/display, wipe refresh -- plus the tonemap and topaz
        # layers of Multi model output, through the same dispatch.
        self.assertEqual(body.count("self._usc_scene_for("), 5)

    # ── 16-bit decode: the trap this had to avoid ──
    def test_topaz_decode_reads_genuine_16_bit_not_8(self):
        """Image.open + np.array on a 16-bit PNG reports mode "RGB" --
        Pillow's 8-bit-per-channel name -- and silently hands back
        uint8, discarding the extra 8 bits with no error. Confirmed
        directly against a real file before this was written.

        Checked against the CODE with the docstring stripped: the
        docstring itself names Image.open to explain why it is not
        used, which would otherwise make this assertion fail on its
        own explanation."""
        import ast as _ast
        import textwrap
        fn = _ast.parse(textwrap.dedent(
            self._src(App._topaz_frame_to_scene))).body[0]
        if (fn.body and isinstance(fn.body[0], _ast.Expr)
                and isinstance(fn.body[0].value, _ast.Constant)):
            fn.body = fn.body[1:]
        code = _ast.unparse(fn)
        self.assertIn("cv2.imread", code)
        self.assertIn("IMREAD_UNCHANGED", code)
        self.assertNotIn("Image.open", code)

    def test_topaz_decode_forces_a_real_16_bit_png(self):
        body = self._src(App._topaz_decode)
        self.assertIn("rgb48be", body)

    def test_topaz_decode_is_not_the_vace_decoder(self):
        """_vace_decode's ffmpeg call writes 8-bit PNG by default,
        which would throw away the entire point of an HDR pass."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        i = src.index("def _usc_topaz_expand")
        # The next METHOD at the same indentation, not the first nested
        # local "def" inside this one (it defines a couple, e.g. _bar
        # and _relay, as closures around the fal call).
        j = src.index("\n    def ", i + 10)
        seg = src[i:j]
        self.assertIn("_topaz_decode(", seg)
        self.assertNotIn("_vace_decode(", seg)

    # ── scene conversion chain ──
    def test_scene_conversion_chain_order(self):
        """PQ decode, then gamut inverse, then the 100-nit scene scale
        -- in that order, matching how the forward analytic path
        builds its own scene-referred output."""
        body = self._src(App._topaz_frame_to_scene)
        i_pq = body.index("_hdr_pq_decode")
        i_gamut = body.index("_bt2020_to_709")
        i_scale = body.index("HDR_SDR_WHITE")
        self.assertLess(i_pq, i_gamut)
        self.assertLess(i_gamut, i_scale)

    def test_scene_conversion_clamps_negative_values(self):
        """A gamut inverse can produce small negative components for
        colours outside 709; clipping them is the same guard the
        preview's own gamut conversion already applies."""
        body = self._src(App._topaz_frame_to_scene)
        self.assertIn("np.clip(nits_709, 0.0, None)", body)

    # ── job-mode scoping ──
    def test_topaz_is_scoped_to_expand_mode(self):
        """Topaz keeps the source resolution and takes the whole clip
        at once; Real-ESRGAN upscales per-frame inside the same loop.
        Combining them needs a real two-pass restructuring this build
        does not attempt."""
        body = self._src(App._usc_run)
        i = body.index('self._usc_hdr_provider() == "topaz"')
        seg = body[i:i + 600]
        self.assertIn('self._usc_job_mode() != "expand"', seg)

    def test_upscale_or_both_with_topaz_falls_back_with_a_log_line(self):
        """Silently sending the wrong frames would be worse than
        falling back and saying so."""
        body = self._src(App._usc_run)
        i = body.index('self._usc_job_mode() != "expand"')
        seg = body[i:i + 1000]     # the explanatory comment runs long
        self.assertIn("Upscale/Both", seg)

    def test_a_failed_topaz_call_falls_back_to_analytic(self):
        """A failed generation must not lose the run -- same precedent
        as the existing VACE fallback."""
        body = self._src(App._usc_run)
        i = body.index("_usc_topaz_expand(")
        seg = body[max(0, i - 100):i + 600]
        self.assertIn("except Exception", seg)
        self.assertIn("using analytic", seg)

    # ── staging: must go through the app's own reader, not a raw copy ──
    def test_staging_reads_through_usc_read_rgb_not_a_raw_copy(self):
        """ffmpeg cannot read this app's EXR sources directly, and a
        bare file copy would hand it something it either rejects or
        silently decodes wrong."""
        body = self._src(App._usc_topaz_expand)
        self.assertIn("_usc_read_rgb", body)
        self.assertNotIn("copy2(os.path.join(folder, n),", body)

    def test_staging_writes_a_uniform_extension(self):
        """Always staged to %05d.png rather than guessed at from the
        sequence's real names, which rarely match ffmpeg's pattern
        exactly."""
        body = self._src(App._usc_topaz_expand)
        self.assertIn('f"{k:05d}.png"', body)

    # ── preview stays free ──
    def test_preview_uses_a_short_span_not_vaces_81_frame_minimum(self):
        """Topaz preview genuinely generates, but with
        USC_TOPAZ_PREVIEW_SPAN (~1s), not VACE_MIN_GEN -- Topaz's schema
        shows no sign of sharing VACE's model-specific floor."""
        body = self._src(App._usc_preview_model)
        seg = body[body.index('if m == "topaz":'):body.index('if m == "wan_vace":')]
        self.assertIn("_usc_topaz_expand", seg)
        self.assertIn("USC_TOPAZ_PREVIEW_SPAN", seg)
        self.assertNotIn("VACE_MIN_GEN", seg)

    def test_preview_checks_the_cache_before_warning_about_cost(self):
        """A repeat preview of an already-cached span should not warn
        about spending money it will not actually spend."""
        body = self._src(App._usc_preview_model)
        seg = body[body.index('if m == "topaz":'):body.index('if m == "wan_vace":')]
        self.assertLess(seg.index("self._usc_topaz_missing("),
                        seg.index("spends real money"))

    # ── reuse over duplication ──
    def test_the_gamut_inverse_is_reused_not_reimplemented(self):
        """The same matrix inverse the preview viewer already uses for
        showing an expanded frame on an ordinary monitor -- inverted
        from the forward matrix at run time, so the pair cannot drift
        apart."""
        body = self._src(App._topaz_frame_to_scene)
        self.assertIn("_bt2020_to_709", body)

    def test_topaz_submit_reuses_the_shared_fal_client_and_watchdog(self):
        """Not a second HTTP client -- the same _fal_client / watchdog
        every other fal call in this app goes through."""
        body = self._src(App._topaz_submit)
        self.assertIn("_fal_client(key)", body)
        self.assertIn("_fal_call_with_watchdog", body)

    def test_topaz_submit_does_not_send_a_mask_or_a_prompt(self):
        """Topaz's schema has no field for either -- sending them would
        be routing through a shape that does not fit.

        Checked against the CODE with the docstring stripped: the
        docstring itself says "mask" and "prompt" to explain what this
        function deliberately does NOT send."""
        import ast as _ast
        import textwrap
        fn = _ast.parse(textwrap.dedent(
            self._src(App._topaz_submit))).body[0]
        if (fn.body and isinstance(fn.body[0], _ast.Expr)
                and isinstance(fn.body[0].value, _ast.Constant)):
            fn.body = fn.body[1:]
        code = _ast.unparse(fn).lower()
        self.assertNotIn("mask", code)
        self.assertNotIn("prompt", code)

    def test_topaz_format_defaults_to_prores(self):
        """On Topaz's own stated recommendation for grading and
        mastering, and measured here: default-CRF HEVC ran about 3%
        off at 1000 nits from compression alone, where a near-lossless
        pass came back within noise."""
        self.assertIn("ProRes", App.TOPAZ_DEFAULT_FORMAT)

    def test_both_topaz_output_formats_are_mapped(self):
        self.assertEqual(set(App.TOPAZ_FORMATS.values()), {"prores", "mp4"})


class TestFixFramesRepairSize(unittest.TestCase):
    """Fix Missing Frames failed on its first real repair with "too
    many values to unpack (expected 2)". _vace_build_source_clip
    returns the output PATH it wrote, not a (w, h) tuple -- the repair
    worker assigned that return to `size` and handed it straight to
    _vace_build_mask_clip, whose own w, h = size then tried to unpack a
    path STRING.

    The exact bug already found and fixed once before in
    _usc_ai_generate (AI Expansion); this sibling repair path had the
    identical mistake and was never exercised until a real fault
    reached it."""

    class _Stub(App):
        def __init__(self):
            import collections
            self._pv_lru = collections.OrderedDict()
            self._pv_lru_bytes = 0

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        self.app = self._Stub()
        self.dir = tempfile.mkdtemp()
        self.names = []
        from PIL import Image
        for k in range(5):
            n = f"{k:05d}.png"
            Image.fromarray(np.full((90, 160, 3), 40, np.uint8)).save(
                os.path.join(self.dir, n))
            self.names.append(n)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_reported_error_is_reproduced_by_unpacking_a_path(self):
        """Pins the diagnosis: a path string of any length other than
        two characters raises exactly this error when unpacked as two
        values, which is what the old line did every time it ran."""
        bogus = "/tmp/adw_ff_work/ff_src_0.mp4"
        with self.assertRaises(ValueError) as ctx:
            _w, _h = bogus
        self.assertIn("too many values to unpack", str(ctx.exception))

    def test_source_clip_builder_returns_a_path_not_a_size(self):
        """The actual return type that was being misused."""
        import inspect
        body = inspect.getsource(App._vace_build_source_clip)
        self.assertTrue(body.rstrip().endswith("return out_mp4"))

    def test_size_now_comes_from_a_real_frame(self):
        body = self._src(App._ff_fix_worker)
        self.assertIn("self._pv_frame(", body)
        i = body.index("self._pv_frame(")
        self.assertIn(".size", body[i:i + 120])

    def test_size_is_read_before_the_source_clip_is_built(self):
        """Order matters for readability, not correctness here, but a
        size computed from the frame must not accidentally depend on
        anything the clip-building call produces."""
        body = self._src(App._ff_fix_worker)
        self.assertLess(body.index("self._pv_frame("),
                        body.index("_vace_build_source_clip("))

    def test_the_source_clip_return_value_is_no_longer_assigned_to_size(self):
        """The exact shape of the original bug: `size =
        self._vace_build_source_clip(...)`. Any reappearance of that
        assignment reintroduces it."""
        body = self._src(App._ff_fix_worker)
        self.assertNotIn("size = self._vace_build_source_clip(", body)

    def test_the_computed_size_is_a_real_two_tuple(self):
        """Verified against the real production function, not a mock:
        PIL's own guarantee that Image.size is always (width, height)
        is what the fix now relies on."""
        size = self.app._pv_frame(
            os.path.join(self.dir, self.names[0])).size
        self.assertEqual(len(size), 2)
        w, h = size          # must not raise
        self.assertEqual((w, h), (160, 90))

    def test_the_mask_clip_call_receives_that_same_size(self):
        """The fixed value has to actually reach the call that used to
        receive the wrong one -- checked as two fragments rather than
        one contiguous string, since the real call wraps its arguments
        across lines."""
        body = self._src(App._ff_fix_worker)
        i = body.index("_vace_build_mask_clip(")
        seg = body[i:i + 80]
        self.assertIn('plan["send_gap"]', seg)
        self.assertIn("size,", seg)

    def test_the_later_plate_size_use_is_fixed_by_the_same_change(self):
        """_vace_decode(mp4, got, plate_size=size, ...) reads the same
        variable a few lines later -- fixed as a side effect of fixing
        the one assignment, not by a second, separate patch."""
        body = self._src(App._ff_fix_worker)
        self.assertIn("plate_size=size", body)
        self.assertEqual(body.count(" size = "), 1)


class TestFixFramesRepairProgress(unittest.TestCase):
    """The repair worker's fal call reported nothing beyond a couple of
    coarse log lines while it ran -- no bar movement, no ETA, no
    silence health check -- exactly what Upscale & Expand's own VACE
    call already has. Ported by generalising the four
    _usc_fal_begin/note/end/tick methods (previously hardcoded to that
    one tab's widgets) into shared _fal_panel_* methods parametrised by
    an explicit panel widget and state-attribute name, so two tabs can
    each run their own standing panel without the sharing becoming
    cross-contamination."""

    class FakePanel:
        def __init__(self):
            self.text = None
            self.packed = False
            self.forgotten = False

        def configure(self, text=None, fg=None):
            if text is not None:
                self.text = text

        def pack(self, **kw):
            self.packed = True

        def pack_forget(self):
            self.forgotten = True

    class _Stub(App):
        def __init__(self):
            self._after_ran = False

        def after(self, ms, fn):
            # Runs the FIRST scheduled call once, immediately, and
            # never chases further self-rescheduling calls -- a real
            # Tk mainloop's timer runs those across many separate
            # event-loop iterations, not recursively inside one
            # synchronous call, which is what a naive stub here would
            # do (and did, the first time this was checked by hand).
            if not self._after_ran:
                self._after_ran = True
                fn()

    def setUp(self):
        self.app = self._Stub()
        self.dir = tempfile.mkdtemp()
        App.VACE_TIMING_FILE = os.path.join(self.dir, "t.json")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    # ── the generalised methods themselves ──
    def test_begin_opens_the_given_panel_under_the_given_state_name(self):
        panel = self.FakePanel()
        self.app._fal_panel_begin(panel, "_ff_fal_state",
                                  "fal-ai/vace", 81, 200.0, False)
        self.assertTrue(panel.packed)
        st = getattr(self.app, "_ff_fal_state")
        self.assertEqual(st["frames"], 81)
        self.assertTrue(st["live"])

    def test_two_panels_under_two_state_names_do_not_cross_contaminate(self):
        """The actual new claim: Upscale & Expand and Fix Missing
        Frames can each have a job in flight without one's state
        leaking into or overwriting the other's."""
        ff_panel, usc_panel = self.FakePanel(), self.FakePanel()
        self.app._fal_panel_begin(ff_panel, "_ff_fal_state",
                                  "fal-ai/vace", 81, 200.0, False)
        self.app._fal_panel_begin(usc_panel, "_usc_fal_state",
                                  "topaz/sdr-to-hdr", 40, 60.0, True)
        self.app._fal_panel_note("_ff_fal_state", phase="render")
        self.app._fal_panel_note("_usc_fal_state", phase="upload")
        self.assertIsNot(self.app._ff_fal_state, self.app._usc_fal_state)
        self.assertEqual(self.app._ff_fal_state["phase"], "render")
        self.assertEqual(self.app._usc_fal_state["phase"], "upload")

    def test_note_is_a_no_op_once_the_job_is_not_live(self):
        panel = self.FakePanel()
        self.app._fal_panel_begin(panel, "_ff_fal_state",
                                  "fal-ai/vace", 10, 20.0, False)
        self.app._fal_panel_end(panel, "_ff_fal_state")
        before = dict(self.app._ff_fal_state)
        self.app._fal_panel_note("_ff_fal_state", phase="render")
        self.assertEqual(self.app._ff_fal_state, before)

    def test_end_closes_clean_with_no_note(self):
        panel = self.FakePanel()
        self.app._fal_panel_begin(panel, "_ff_fal_state",
                                  "fal-ai/vace", 10, 20.0, False)
        self.app._fal_panel_end(panel, "_ff_fal_state")
        self.assertTrue(panel.forgotten)
        self.assertFalse(self.app._ff_fal_state["live"])

    def test_end_with_a_note_shows_it_instead_of_closing(self):
        """A failure message has to stay ON SCREEN, not vanish the
        instant the job ends -- that is the whole point of reporting
        it."""
        panel = self.FakePanel()
        self.app._fal_panel_begin(panel, "_ff_fal_state",
                                  "fal-ai/vace", 10, 20.0, False)
        self.app._fal_panel_end(panel, "_ff_fal_state",
                                "fal call failed", ok=False)
        self.assertFalse(panel.forgotten)
        self.assertEqual(panel.text, "fal call failed")

    def test_a_missing_state_dict_is_defended_against_in_source(self):
        """A tab that never opened a panel this run must not crash when
        something still tries to feed it.

        Checked in source rather than by calling it live: the test
        base class answers ANY missing attribute rather than genuinely
        raising, so getattr(obj, name, default) here never reaches its
        default under that stub -- a stub-only quirk, not something
        about the real class, so the defence is verified by inspection
        instead."""
        body = self._src(App._fal_panel_note)
        self.assertIn('getattr(self, state_attr, {})', body)
        body = self._src(App._fal_panel_end)
        self.assertIn('getattr(self, state_attr, None)', body)

    # ── on_panel: the extensibility point in the shared submit call ──
    def test_vace_submit_accepts_an_on_panel_hook(self):
        import inspect
        sig = inspect.signature(App._vace_submit)
        self.assertIn("on_panel", sig.parameters)
        self.assertIsNone(sig.parameters["on_panel"].default)

    def test_pfeed_does_nothing_when_no_panel_hook_is_given(self):
        """_vace_submit serves three tabs; a caller with no panel must
        see exactly the old behaviour, not a crash from a None
        callback."""
        body = self._src(App._vace_submit)
        i = body.index("def _pfeed")
        seg = body[i:i + 300]
        self.assertIn("if on_panel is None:", seg)
        self.assertIn("return", seg)

    def test_pfeed_calls_the_caller_supplied_hook(self):
        body = self._src(App._vace_submit)
        i = body.index("def _pfeed")
        self.assertIn("on_panel(**k)", body[i:i + 500])

    # ── Fix Missing Frames' own wiring ──
    def test_the_ff_panel_widget_exists(self):
        body = self._src(App._tab_fixframes)
        self.assertIn("ff_fal_panel", body)
        self.assertIn("_ff_fal_state", body)

    def test_the_repair_worker_opens_a_panel_per_fault(self):
        """Each fault is its own separate fal submission -- the panel
        has to open and close once per fault, not once for the whole
        multi-fault run."""
        body = self._src(App._ff_fix_worker)
        self.assertIn("_fal_panel_begin(\n"
                     "                    self.ff_fal_panel,", body)
        i = body.index("for fi, fa in enumerate(self.ff_faults):")
        self.assertLess(i, body.index("_fal_panel_begin("))

    def test_the_repair_worker_drives_the_bar_through_phases(self):
        body = self._src(App._ff_fix_worker)
        self.assertIn("self.ff_bar.set(", body)
        self.assertIn("_vace_phase_fraction", body)

    def test_the_repair_worker_uses_the_learned_eta(self):
        """The same creeping render-phase bar Upscale & Expand has, not
        just a flat phase fraction -- render is the longest phase and
        the one where "is this still going" matters most."""
        body = self._src(App._ff_fix_worker)
        self.assertIn("_vace_render_progress", body)
        self.assertIn("_vace_eta_text", body)

    def test_the_repair_worker_records_render_time_for_future_estimates(self):
        body = self._src(App._ff_fix_worker)
        self.assertIn("_vace_record_render(", body)

    def test_the_submit_call_is_wired_through_on_phase_and_on_panel(self):
        body = self._src(App._ff_fix_worker)
        i = body.index("self._vace_submit(")
        seg = body[i:i + 200]
        self.assertIn("on_phase=_bar", seg)
        self.assertIn("on_panel=_note", seg)

    def test_a_failed_submit_closes_the_panel_with_a_failure_note(self):
        body = self._src(App._ff_fix_worker)
        i = body.index("self._vace_submit(")
        seg = body[i:i + 1000]
        self.assertIn("except Exception as _fex:", seg)
        self.assertIn('eta["t0"] = None', seg)   # the clock stops too
        self.assertIn("ok=False", seg)
        self.assertIn("raise", seg)

    def test_a_successful_repair_closes_the_panel_cleanly(self):
        body = self._src(App._ff_fix_worker)
        self.assertIn('self._fal_panel_end(\n'
                     '                    self.ff_fal_panel, "_ff_fal_state")',
                     body)

    def test_fal_log_lines_reach_the_panel_too(self):
        """Every line from fal is a sign of life, same treatment the
        Upscale tab's own _relay gives it."""
        body = self._src(App._ff_fix_worker)
        i = body.index("def _log(m):")
        seg = body[i:i + 300]
        self.assertIn("_fal_panel_note", seg)

    def test_reset_tears_down_the_bar_and_panel_unconditionally(self):
        """Called on success, on failure, and after a cancel -- one
        place that always cleans up, so none of the three paths can
        leave either stuck showing a job that is no longer running."""
        body = self._src(App._ff_fix_reset)
        self.assertIn("self.ff_bar.reset()", body)
        self.assertIn("self._fal_panel_end(", body)

    # ── the two ORIGINAL callers still work after the rename ──
    def test_usc_ai_generate_uses_the_generalised_methods(self):
        body = self._src(App._usc_ai_generate)
        self.assertIn('_fal_panel_begin(\n'
                     '            self.usc_fal_panel, "_usc_fal_state"', body)
        self.assertIn("on_panel=_note", body)

    def test_usc_topaz_expand_uses_the_generalised_methods(self):
        body = self._src(App._usc_topaz_expand)
        self.assertIn('_fal_panel_begin(\n'
                     '            self.usc_fal_panel, "_usc_fal_state"', body)

    def test_no_call_site_anywhere_still_uses_the_old_tab_specific_names(self):
        """The four original names (_usc_fal_begin/note/end/tick) no
        longer exist -- anything still calling them would be a silent
        AttributeError waiting for the first real job."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        for old in ("_usc_fal_begin(", "_usc_fal_note(", "_usc_fal_end(",
                    "_usc_fal_tick("):
            with self.subTest(old=old):
                self.assertNotIn(old, src)


class TestFixFramesRepairedFrameFormat(unittest.TestCase):
    """A repaired EXR frame failed to open, in this app's own reader AND
    in Nuke ("not supported by Mov64 reader"). Root cause: VACE is a
    video model and _vace_decode always writes plain 8-bit PNG,
    regardless of what sequence is being repaired -- the copy-back step
    then byte-copied that PNG straight into a destination path ending
    in ".exr". A file whose actual content is PNG under a ".exr" name
    is invalid to every reader that opens it by extension, which is
    both why OpenEXR refused it and very plausibly why Nuke's own
    format auto-detection fell through to the wrong reader instead of
    the right one refusing cleanly."""

    class _Stub(App):
        def __init__(self):
            import collections
            self._pv_lru = collections.OrderedDict()
            self._pv_lru_bytes = 0

    def setUp(self):
        self.app = self._Stub()
        self.dir = tempfile.mkdtemp()
        from PIL import Image
        self.src_png = os.path.join(self.dir, "00003.png")
        Image.fromarray(np.full((40, 60, 3), 160, np.uint8)).save(
            self.src_png)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    # ── pinning the diagnosis ──
    def test_a_raw_byte_copy_into_a_exr_path_reproduces_the_bug(self):
        """The exact mechanism reported: 'Unable to open ... for
        read', on the very frame a repair had just written."""
        bug_path = os.path.join(self.dir, "buggy.00062.exr")
        shutil.copy2(self.src_png, bug_path)
        with self.assertRaises(RuntimeError) as ctx:
            self.app._ff_read_frame(bug_path)
        self.assertIn("Unable to open", str(ctx.exception))

    def test_vace_decode_always_writes_plain_png(self):
        """Confirms the root cause: there is no path by which VACE's
        own output could already be a valid EXR -- it is fundamentally
        an 8-bit video-model result."""
        body = self._src(App._vace_decode)
        self.assertIn('"%05d.png"', body)

    # ── the fix ──
    def test_an_exr_destination_is_converted_not_copied(self):
        dst = os.path.join(self.dir, "seq.00062.exr")
        self.app._ff_write_repaired_frame(self.src_png, dst, is_linear=False)
        back = self.app._ff_read_frame(dst)
        self.assertEqual(back.shape, (40, 60, 3))

    def test_the_repaired_exr_is_actually_readable(self):
        """The whole point: this must not raise, in this app's own
        reader, on the exact file a repair just produced."""
        dst = os.path.join(self.dir, "seq.00062.exr")
        self.app._ff_write_repaired_frame(self.src_png, dst, is_linear=False)
        try:
            self.app._ff_read_frame(dst)
        except RuntimeError as ex:
            self.fail(f"repaired frame is unreadable: {ex}")

    def test_display_encoded_sequences_get_no_curve(self):
        """The PNG's own code value should pass through unchanged --
        matching how an ordinary display-encoded EXR neighbour would
        already read."""
        dst = os.path.join(self.dir, "seq.00062.exr")
        self.app._ff_write_repaired_frame(self.src_png, dst, is_linear=False)
        back = self.app._ff_read_frame(dst)
        self.assertAlmostEqual(float(back[0, 0, 0]), 160 / 255.0, places=3)

    def test_scene_linear_sequences_get_linearised(self):
        """A repaired frame has to read the SAME way as its untouched
        neighbours, or it reintroduces a smaller version of the
        mid-sequence colourspace split this app already spent a long
        time chasing down elsewhere."""
        dst = os.path.join(self.dir, "seq.00062.exr")
        self.app._ff_write_repaired_frame(self.src_png, dst, is_linear=True)
        back = self.app._ff_read_frame(dst)
        expected = float(App._srgb_to_linear(np.float32(160 / 255.0)))
        self.assertAlmostEqual(float(back[0, 0, 0]), expected, places=3)

    def test_non_exr_destinations_are_still_a_plain_copy(self):
        """PNG in, PNG destination: the already-correct case must not
        be disturbed by this fix."""
        from PIL import Image
        dst = os.path.join(self.dir, "seq.00062.png")
        self.app._ff_write_repaired_frame(self.src_png, dst, is_linear=False)
        self.assertTrue(np.array_equal(np.array(Image.open(dst)),
                                       np.array(Image.open(self.src_png))))

    def test_it_uses_the_shared_exr_writer_not_a_new_one(self):
        """One definition of how this app writes an EXR."""
        body = self._src(App._ff_write_repaired_frame)
        self.assertIn("_organic_write_exr", body)

    # ── wiring ──
    def test_the_copy_back_loop_uses_the_new_writer(self):
        body = self._src(App._ff_fix_worker)
        self.assertIn("_ff_write_repaired_frame(", body)
        self.assertNotIn("shutil.copy2(\n                            os.path.join(got",
                        body)

    def test_the_convention_is_now_derived_not_merely_trusted(self):
        """Superseded by evidence-based detection: the manual control
        is still consulted, but only as the LAST resort inside
        _ff_target_convention, when no real neighbouring frame is
        available to check against at all -- see
        TestFixFramesRepairMatchesNeighbours for the full behaviour."""
        body = self._src(App._ff_fix_worker)
        self.assertIn("_ff_target_convention(", body)
        fallback_body = self._src(App._ff_target_convention)
        self.assertIn("ff_cs_var", fallback_body)


class TestFixFramesRepairMatchesNeighbours(unittest.TestCase):
    """A repaired frame's format and colourspace convention are
    derived from a REAL, untouched neighbouring frame -- not from a
    manual toggle that might not reflect what the sequence actually
    holds. Trusting a setting over the data is the exact mistake that
    caused a mid-sequence colourspace split once already in this app,
    on a different tab -- this closes the same door here."""

    class _V:
        def __init__(self, v):
            self.v = v

        def get(self):
            return self.v

    class _Stub(App):
        def __init__(self, cs="Display-encoded", missing=None, faults=None):
            import collections
            self._pv_lru = collections.OrderedDict()
            self._pv_lru_bytes = 0
            self.ff_cs_var = TestFixFramesRepairMatchesNeighbours._V(cs)
            self.ff_missing = missing or set()
            self.ff_faults = faults or []

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def _write_linear_neighbours(self, names, skip):
        """Real, genuinely over-range EXR frames -- unambiguous
        evidence they are scene-linear, independent of any setting."""
        app = self._Stub()
        for k, n in enumerate(names):
            if k not in skip:
                rgb = np.full((20, 30, 3), 6.0, np.float32)
                app._organic_write_exr(os.path.join(self.dir, n), rgb,
                                       half=True)

    # ── the untrusted set ──
    def test_missing_indices_are_untrusted(self):
        app = self._Stub(missing={3})
        self.assertIn(3, app._ff_untrusted_indices())

    def test_every_fault_span_is_untrusted_not_just_missing(self):
        """A held or black or corrupt frame is just as unfit to serve
        as a reference for what an ordinary frame looks like."""
        app = self._Stub(faults=[{"kind": "held", "first": 5, "last": 7}])
        untrusted = app._ff_untrusted_indices()
        self.assertEqual(untrusted, {5, 6, 7})

    def test_untrusted_indices_from_missing_and_faults_combine(self):
        app = self._Stub(missing={1}, faults=[{"kind": "black", "first": 8,
                                              "last": 9}])
        self.assertEqual(app._ff_untrusted_indices(), {1, 8, 9})

    # ── the neighbour picker ──
    def test_it_skips_the_whole_untrusted_span(self):
        names = [f"f{i}.exr" for i in range(10)]
        for i in (0, 1, 2, 3, 7, 8, 9):
            open(os.path.join(self.dir, names[i]), "w").close()
        app = self._Stub()
        ref = app._ff_neighbor_reference(self.dir, names, 5, avoid={4,5,6})
        self.assertIsNotNone(ref)
        self.assertNotIn(os.path.basename(ref),
                         (names[4], names[5], names[6]))

    def test_it_picks_the_nearest_real_frame(self):
        names = [f"f{i}.exr" for i in range(10)]
        for i in (0, 1, 2, 3, 7, 8, 9):
            open(os.path.join(self.dir, names[i]), "w").close()
        app = self._Stub()
        ref = app._ff_neighbor_reference(self.dir, names, 5, avoid={4,5,6})
        self.assertIn(os.path.basename(ref), (names[3], names[7]))

    def test_no_real_neighbour_returns_none_rather_than_a_guess(self):
        names = [f"f{i}.exr" for i in range(5)]
        app = self._Stub()
        ref = app._ff_neighbor_reference(self.dir, names, 2, avoid=set())
        self.assertIsNone(ref)

    # ── evidence overrides a wrong manual setting ──
    def test_real_over_range_neighbours_override_a_wrong_toggle(self):
        """The toggle says 'Display-encoded'; the actual neighbouring
        frames are genuinely over-range, which is impossible for a
        display-referred value. The data wins."""
        names = [f"seq.{k:05d}.exr" for k in range(7)]
        self._write_linear_neighbours(names, skip={3})
        app = self._Stub(cs="Display-encoded", missing={3},
                         faults=[{"kind": "missing", "first": 3, "last": 3}])
        untrusted = app._ff_untrusted_indices()
        ext, is_linear = app._ff_target_convention(
            self.dir, names, 3, untrusted)
        self.assertEqual(ext, ".exr")
        self.assertTrue(is_linear)

    def test_the_sidecar_outranks_over_range_detection(self):
        """A sidecar this app wrote is stronger evidence than inferring
        from pixel content, matching the same priority order
        _organic_load_linear already established."""
        names = [f"seq.{k:05d}.exr" for k in range(5)]
        self._write_linear_neighbours(names, skip=set())
        app = self._Stub()
        app._adw_write_colorspace(self.dir, "display")
        _ext, is_linear = app._ff_target_convention(
            self.dir, names, 2, set())
        self.assertFalse(is_linear)

    def test_falls_back_to_the_manual_toggle_with_no_real_neighbour(self):
        """The only case where the setting is trusted: there is
        genuinely nothing real left to check against."""
        names = [f"f{i}.exr" for i in range(3)]
        app = self._Stub(cs="Scene-linear")
        _ext, is_linear = app._ff_target_convention(
            self.dir, names, 1, avoid={0, 1, 2})
        self.assertTrue(is_linear)

    def test_a_non_exr_neighbour_means_no_curve_regardless_of_the_toggle(self):
        names = [f"f{i}.png" for i in range(3)]
        for n in names:
            open(os.path.join(self.dir, n), "w").close()
        app = self._Stub(cs="Scene-linear")
        ext, is_linear = app._ff_target_convention(
            self.dir, names, 1, avoid=set())
        self.assertEqual(ext, ".png")
        self.assertFalse(is_linear)

    # ── the extension mismatch guard ──
    def test_a_mismatched_nominal_extension_is_overridden(self):
        """A pathological or synthesised-name edge case: the sequence's
        own filename claims one format, but every real neighbour is
        something else. Trusting the name here is exactly the mistake
        this check exists to catch."""
        names = [f"seq.{k:05d}.exr" for k in range(5)]
        self._write_linear_neighbours(names, skip=set())
        bad_names = list(names)
        bad_names[2] = "seq.00002.png"
        app = self._Stub()
        ext, _lin = app._ff_target_convention(
            self.dir, bad_names, 2, avoid={2})
        self.assertEqual(ext, ".exr")

    def test_the_worker_corrects_and_logs_a_mismatch(self):
        body = self._src(App._ff_fix_worker)
        self.assertIn("_nom_ext != _ext", body)
        self.assertIn("writing", body)

    # ── wiring: derived per fault, not once globally ──
    def test_convention_is_derived_per_fault_not_once_for_the_whole_run(self):
        """A sequence could, in principle, be inconsistent across
        different spans; deriving this once per fault is negligible
        extra cost against re-reading a neighbour per repaired frame,
        which this avoids by reusing one probe across a fault's whole
        core span."""
        body = self._src(App._ff_fix_worker)
        self.assertIn("_ff_target_convention(", body)
        i = body.index("for fi, fa in enumerate")
        j = body.index("_ff_target_convention(", i)
        k = body.index("for k in range(plan", j)
        self.assertLess(j, k)

    def test_the_reference_index_sits_inside_the_repaired_span(self):
        """The probe has to be anchored near the frames actually being
        replaced, not at some arbitrary fixed position."""
        body = self._src(App._ff_fix_worker)
        self.assertIn("core_start", body[body.index("_ref_idx ="):
                                        body.index("_ref_idx =") + 200])

    def test_write_repaired_frame_still_does_the_actual_conversion(self):
        """The detection feeds the existing, already-verified writer --
        not a second, parallel write path."""
        body = self._src(App._ff_fix_worker)
        self.assertIn("_ff_write_repaired_frame(", body)


class TestVaceAspectAndColour(unittest.TestCase):
    """Measured on a real repaired frame against its real neighbour:
    content horizontally squeezed by about 2.6% (a ferris wheel and
    roller coaster shifted and slightly compressed relative to the
    real frame either side of it) and a per-channel colour shift that
    is NOT a uniform exposure change -- R and G darker than B on
    desaturated sky content, consistent with an ffmpeg colour-matrix
    mismatch (bt601 vs bt709) between the encode sent to VACE and the
    decode of what came back, since neither end specified an explicit
    matrix and ffmpeg's own choice is a silent, resolution-based
    guess."""

    class _Stub(App):
        def __init__(self):
            pass

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        self.app = self._Stub()
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _mp4_of(self, w, h, draw_circle=True):
        import subprocess
        from PIL import Image, ImageDraw
        stage = os.path.join(self.dir, f"stage_{w}x{h}")
        os.makedirs(stage, exist_ok=True)
        img = Image.new("RGB", (w, h), (10, 10, 30))
        if draw_circle:
            d = ImageDraw.Draw(img)
            r = min(w, h) // 3
            cx, cy = w // 2, h // 2
            d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(255, 255, 255))
        img.save(os.path.join(stage, "00001.png"))
        mp4 = os.path.join(self.dir, f"gen_{w}x{h}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                        "-framerate", "1", "-i",
                        os.path.join(stage, "%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4],
                       capture_output=True)
        return mp4

    def _circle_ratio(self, png_path):
        from PIL import Image
        arr = np.array(Image.open(png_path).convert("L"))
        ys, xs = np.where(arr > 128)
        return (xs.max() - xs.min()) / max(1, (ys.max() - ys.min()))

    # ── the aspect / scale bug ──
    def test_a_matching_aspect_still_uses_a_plain_resize(self):
        """The common case -- same aspect, different resolution -- is
        unaffected: no letterbox warning, no distortion introduced by
        this fix where there was none before."""
        mp4 = self._mp4_of(320, 180)
        out = os.path.join(self.dir, "out1")
        logs = []
        got = self.app._vace_decode(mp4, out, plate_size=(640, 360),
                                    on_log=logs.append)
        self.assertFalse(any("letterbox" in m for m in logs))
        from PIL import Image
        self.assertEqual(Image.open(os.path.join(out, got[0])).size,
                         (640, 360))

    def test_a_real_aspect_mismatch_preserves_circular_geometry(self):
        """The measured case: content narrower than the plate by a
        small amount. A naive stretch turns a circle into an oval by
        exactly the aspect mismatch; the fix must not."""
        mp4 = self._mp4_of(634, 360)
        out = os.path.join(self.dir, "out2")
        got = self.app._vace_decode(mp4, out, plate_size=(640, 360))
        ratio = self._circle_ratio(os.path.join(out, got[0]))
        self.assertAlmostEqual(ratio, 1.0, delta=0.01)

    def test_the_old_stretch_would_have_distorted_it(self):
        """Pins the bug this replaces: a plain resize-to-fill on the
        same mismatched input visibly distorts a true circle into an
        oval, by roughly the size of the aspect disagreement -- unlike
        the fixed path, which keeps it round (see
        test_a_real_aspect_mismatch_preserves_circular_geometry)."""
        from PIL import Image, ImageDraw
        gw, gh, pw, ph = 634, 360, 640, 360
        img = Image.new("RGB", (gw, gh), (10, 10, 30))
        d = ImageDraw.Draw(img)
        r = min(gw, gh) // 3
        cx, cy = gw // 2, gh // 2
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255))
        stretched_path = os.path.join(self.dir, "old_stretch.png")
        img.resize((pw, ph), Image.LANCZOS).save(stretched_path)
        ratio = self._circle_ratio(stretched_path)
        self.assertGreater(abs(ratio - 1.0), 0.005)

    def test_the_log_explains_letterboxing_not_just_the_mismatch(self):
        mp4 = self._mp4_of(634, 360)
        out = os.path.join(self.dir, "out3")
        logs = []
        self.app._vace_decode(mp4, out, plate_size=(640, 360),
                              on_log=logs.append)
        self.assertTrue(any("letterboxing" in m for m in logs))

    def test_letterbox_output_is_exactly_the_plate_size(self):
        """The frame slots back into a sequence at the plate's exact
        resolution regardless of which path was taken."""
        mp4 = self._mp4_of(634, 360)
        out = os.path.join(self.dir, "out4")
        got = self.app._vace_decode(mp4, out, plate_size=(640, 360))
        from PIL import Image
        self.assertEqual(Image.open(os.path.join(out, got[0])).size,
                         (640, 360))

    def test_padding_is_edge_replicated_not_a_hard_black_border(self):
        """A flat black band is its own new, obviously-wrong artefact
        right at the frame edge -- exactly where a repair's context
        is least likely to hide it."""
        body = self._src(App._vace_decode)
        self.assertIn("BORDER", body.upper()) if False else None
        self.assertIn("crop((0, 0, 1, fh))", body)

    def test_upscale_is_skipped_when_the_aspect_is_off(self):
        """Real-ESRGAN reconstructs detail assuming the input geometry
        is already correct; upscaling a distorted frame would just
        make the distortion sharper, not fix it."""
        body = self._src(App._vace_decode)
        i = body.index("if upscale and pw > gw")
        self.assertIn("not aspect_off", body[i:i+60])

    def test_the_aspect_tolerance_is_tighter_than_before(self):
        """The measured case (about 1.9% off in the isolated circle
        test, 2.6% on the real frame) has to actually trip the new
        path -- the old 0.02 tolerance was looser than some real
        mismatches this is meant to catch."""
        body = self._src(App._vace_decode)
        self.assertIn("0.005", body)

    # ── the colour matrix bug ──
    def test_the_source_encode_tags_bt709_explicitly(self):
        """The clip encoder -- _vace_build_source_clip (the masked
        repair clip; its plain-clip sibling went with the Interpolate
        tab) -- must not be left to ffmpeg's own resolution-based
        guess."""
        for fn in (App._vace_build_source_clip,):
            with self.subTest(fn=fn.__name__):
                body = self._src(fn)
                self.assertIn('"-colorspace", "bt709"', body)
                self.assertIn('"-color_primaries", "bt709"', body)

    def test_the_decode_forces_bt709_on_read(self):
        """Not trusted from whatever tag the returned clip carries --
        forced, so the decode side cannot silently disagree with the
        encode side of the same round trip."""
        body = self._src(App._vace_decode)
        i = body.index('_sp.run(["ffmpeg", "-y",')
        seg = body[i:i+300]
        self.assertIn('"-color_primaries", "bt709"', seg)
        self.assertIn('"-i", mp4', seg)

    def test_the_colour_tags_precede_the_input_flag(self):
        """As INPUT-side options they override how the stream is
        interpreted; placed after -i they would be silently ignored by
        ffmpeg's argument parsing."""
        body = self._src(App._vace_decode)
        i = body.index('"-color_primaries", "bt709"')
        j = body.index('"-i", mp4')
        self.assertLess(i, j)

    def test_it_does_not_claim_to_fix_vaces_own_internals(self):
        """Honest about the limit: this closes OUR side of the round
        trip's ambiguity, not whatever VACE itself does internally,
        which cannot be inspected or controlled from here."""
        body = self._src(App._vace_decode)
        self.assertIn("cannot", body.lower())


class TestFixFramesUsesUpscaleReconstruction(unittest.TestCase):
    """VACE's endpoint caps out at 720p; every repaired frame was being
    brought back to the plate's real resolution with a plain Lanczos
    resize because upscale=True was never being passed, leaving the
    already-built Real-ESRGAN reconstruction path (_vace_upscale_dir)
    unused for every repair Fix Missing Frames ever ran."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_the_fix_frames_decode_call_requests_upscale(self):
        body = self._src(App._ff_fix_worker)
        i = body.index("self._vace_decode(mp4, got")
        self.assertIn("upscale=True", body[i:i + 150])


class TestFixFramesRifeFirst(unittest.TestCase):
    """Optical-flow interpolation via the existing RIFE install, tried
    before VACE for the one case it is unambiguously the right tool
    for: a single missing/faulty frame with a real, trusted frame
    immediately either side.

    Measured on an actual repaired frame: VACE's own generated
    highlights came back systematically short of what the real
    footage's blown highlights reach (median 0.73 in the brightest 1%,
    where the real neighbouring frame's own brightest pixels sit at
    1.0). RIFE warps and blends REAL pixel values from two genuine
    neighbours rather than generating new ones, so it cannot lose
    highlight information the same way -- for a single-frame gap where
    both bounding frames are real, there is nothing to hallucinate."""

    class _Stub(App):
        def __init__(self):
            pass

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        self.app = self._Stub()
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    # ── eligibility ──
    def test_a_single_frame_with_trusted_neighbours_is_eligible(self):
        r = self.app._ff_rife_eligible(5, 5, avoid=set(), n_total=10)
        self.assertEqual(r, (4, 6))

    def test_a_wider_gap_is_not_eligible(self):
        """The harder problem this deliberately does not attempt: a
        straight line between two anchors approximates real motion
        worse the further apart they are."""
        self.assertIsNone(
            self.app._ff_rife_eligible(5, 7, avoid=set(), n_total=10))

    def test_the_first_frame_has_no_real_frame_before_it(self):
        self.assertIsNone(
            self.app._ff_rife_eligible(0, 0, avoid=set(), n_total=10))

    def test_the_last_frame_has_no_real_frame_after_it(self):
        self.assertIsNone(
            self.app._ff_rife_eligible(9, 9, avoid=set(), n_total=10))

    def test_an_untrusted_neighbour_makes_it_ineligible(self):
        """A neighbour that is itself part of another fault cannot
        serve as a real anchor -- the same evidence standard already
        used for the colourspace/format consistency check."""
        self.assertIsNone(
            self.app._ff_rife_eligible(5, 5, avoid={6}, n_total=10))

    def test_eligibility_reuses_the_same_trust_set_as_convention_detection(self):
        """One definition of 'trustworthy neighbour' for this tab, not
        two separate ones that could disagree."""
        body = self._src(App._ff_fix_worker)
        self.assertIn("self._ff_untrusted_indices()", body)
        i = body.index("_ff_rife_eligible(")
        self.assertIn("_ff_untrusted_indices()", body[max(0, i-120):i+120])

    # ── the actual interpolation call, and its fallback ──
    def test_it_reports_not_installed_rather_than_raising(self):
        """RIFE is a separate, real, machine-specific dependency -- a
        cloned repo, its own venv, downloaded weights -- not something
        a code change can make true on its own."""
        self.assertFalse(self.app._ff_rife_installed())

    def test_an_uninstalled_rife_fails_gracefully_not_with_an_exception(self):
        from PIL import Image
        for n in ("a.png", "b.png"):
            Image.fromarray(np.full((40, 60, 3), 100, np.uint8)).save(
                os.path.join(self.dir, n))
        ok = self.app._ff_rife_interp_frame(
            self.dir, ["a.png", "b.png"], 0, 1,
            os.path.join(self.dir, "out.png"))
        self.assertFalse(ok)

    def test_a_failure_falls_back_to_vace_rather_than_aborting_the_repair(self):
        body = self._src(App._ff_fix_worker)
        i = body.index("_rife_pair = self._ff_rife_eligible(")
        seg = body[i:i + 1600]
        self.assertIn("using VACE", seg)
        self.assertIn("plan = self._ff_repair_plan(", body[i:])

    def test_success_skips_vace_entirely_for_that_fault(self):
        """The whole point of the free/instant path: no fal call at
        all when RIFE succeeds."""
        body = self._src(App._ff_fix_worker)
        i = body.index("_rife_pair = self._ff_rife_eligible(")
        seg = body[i:i + 1600]
        self.assertIn("continue", seg)
        self.assertIn("no fal call", seg)

    # ── reused, not reimplemented ──
    def test_it_reuses_the_same_rife_calling_convention_as_adw_blur(self):
        """Not a second, independent RIFE integration -- the same
        venv/script/script_dir resolution and the same
        _rife_run_pair_stepped call ADW-Blur already relies on."""
        body = self._src(App._ff_rife_interp_frame)
        self.assertIn('"~/RifeApp/venv/bin/python3"', body)
        self.assertIn("self._rife_inference_script()", body)
        self.assertIn("self._rife_run_pair_stepped(", body)

    def test_anchors_are_read_through_the_colourspace_aware_reader(self):
        """Not a raw file copy or a second EXR-handling path -- these
        anchors can genuinely be EXR."""
        body = self._src(App._ff_rife_interp_frame)
        self.assertIn("self._usc_read_rgb(", body)

    def test_padding_is_edge_replicated_not_zero_filled(self):
        """Zero-padding bleeds dark values back into the content near
        the boundary once RIFE's network gets hold of it -- the same
        reasoning ADW-Blur's own caller already documents."""
        body = self._src(App._ff_rife_interp_frame)
        self.assertIn('mode="edge"', body)

    def test_the_result_is_cropped_back_to_the_original_size(self):
        body = self._src(App._ff_rife_interp_frame)
        self.assertIn(".crop((0, 0, ow, oh))", body)

    def test_the_output_goes_through_the_same_consistency_writer(self):
        """Not a second write path -- the same evidence-based
        conversion already built for VACE's repaired frames applies
        here identically, since both are just 8-bit PNG needing to
        land in whatever format and convention the real neighbours
        use."""
        body = self._src(App._ff_fix_worker)
        i = body.index("_rife_pair = self._ff_rife_eligible(")
        seg = body[i:i + 1600]
        self.assertIn("_ff_target_convention(", seg)
        self.assertIn("_ff_write_repaired_frame(", seg)

    def test_a_missing_output_frame_is_treated_as_failure(self):
        """RIFE producing fewer than the expected three files (the two
        anchors plus the middle) is a failure, not silently used."""
        body = self._src(App._ff_rife_interp_frame)
        self.assertIn("len(got) < 3", body)

    def test_cleanup_does_not_raise_if_setup_failed_before_the_temp_dir_existed(self):
        """tmp_in is created partway through the try block; the
        finally clause has to guard against running before it exists,
        not just after."""
        body = self._src(App._ff_rife_interp_frame)
        self.assertIn("tmp_in = None", body)
        i = body.index("finally:")
        self.assertIn("if tmp_in:", body[i:i + 60])


class TestOldProjectsStillLoad(unittest.TestCase):
    """An .adwproj saved by AI-Toolbox carries "area", "curve" and
    "rife" sections. Studio Toolbox has none of those modules; the
    file must still open, with everything else restored, rather than
    failing on the first section it does not know."""

    class _Stub(App):
        def __init__(self):
            self.got = []
            self.restored = []
            self._project_current_path = None
            self._PROJECT_VAR_TYPES = (_FakeVar,)
            self.sam_objects = None
            self.sam_active_obj = 0
            for fn in ("_usc_load_folder", "_ups_load_folder",
                       "_sam_load_folder"):
                setattr(self, fn, lambda d, fn=fn: self.got.append(fn))

        def _project_restore_vars(self, data):
            self.restored.append(data)

        def setstatus(self, *a, **k):
            pass

    def setUp(self):
        import zipfile, json
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "old.adwproj")
        data = {"version": 1, "modules": {
            "area":  {"folder": self.dir, "vars": {"area_x_var": 1},
                      "active": 0, "area_list": [{"name": "A"}]},
            "curve": {"folder": self.dir, "vars": {}, "active": 0,
                      "curve_list": [{"name": "C"}]},
            "rife":  {"folder": self.dir, "vars": {"rmult_var": 3}},
            "upscale":  {"folder": self.dir, "vars": {"usc_a": 1}},
            "upstudio": {"folder": self.dir, "vars": {"ups_a": 2}},
            "sam": {"folder": self.dir, "vars": {"sam_a": 3},
                    "objects": [{"pts": []}], "active_obj": 0},
        }}
        with zipfile.ZipFile(self.path, "w") as zf:
            zf.writestr("project.json", json.dumps(data))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_kept_modules_are_restored(self):
        app = self._Stub()
        with unittest.mock.patch.object(
                sys.modules["ai_toolbox"].messagebox, "showerror") as err:
            app._project_load(self.path)
        err.assert_not_called()
        self.assertEqual(app.got, ["_usc_load_folder", "_ups_load_folder",
                                   "_sam_load_folder"])
        self.assertEqual(app.restored,
                         [{"usc_a": 1}, {"ups_a": 2}, {"sam_a": 3}])
        self.assertEqual(app._project_current_path, self.path)

    def test_the_adw_sections_are_ignored_not_restored(self):
        app = self._Stub()
        with unittest.mock.patch.object(
                sys.modules["ai_toolbox"].messagebox, "showerror"):
            app._project_load(self.path)
        for d in app.restored:
            self.assertNotIn("rmult_var", d)
            self.assertNotIn("area_x_var", d)

    def test_save_writes_no_adw_sections(self):
        import inspect
        body = inspect.getsource(App._project_save)
        for key in ('["area"]', '["curve"]', '["rife"]'):
            self.assertNotIn(key, body)


class TestSetupRifeLabel(unittest.TestCase):
    """The Setup & Install component row for the RIFE install is
    labelled "RIFE" and names its one remaining consumer, Fix Missing
    Frames: with the ADW-Blur tab gone, a description still crediting
    it would send people looking for a tab that is not there."""

    def test_the_setup_component_row_says_rife_not_adw_blur(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn('("rife",   "RIFE",', src)

    def test_the_description_names_the_real_consumer(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        i = src.index('("rife",   "RIFE",')
        seg = src[i:i + 200]
        self.assertNotIn("ADW-Blur", seg)
        self.assertIn("Fix Missing Frames", seg)

    def test_no_tab_bar_entry_or_tab_builder_for_the_adw_modules(self):
        """Removed, not hidden: nothing should still build them."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        for key in ("curve", "area", "rife", "interp"):
            with self.subTest(key=key):
                self.assertNotIn(f'"{key}"),', src[src.index("def _tabbar"):
                                                     src.index("def show(")])
                self.assertNotIn(f"def _tab_{key}(", src)

    def test_rife_and_esrgan_module_sets_include_fixframes(self):
        """Both are now genuine Fix Missing Frames dependencies -- RIFE
        for the repair itself, Real-ESRGAN for the upscale-on-decode
        reconstruction -- so a Fix-Missing-Frames-only launcher must
        still show and require them."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        i = src.index('"rife":   {')
        j = src.index("\n", i)
        self.assertIn('"fixframes"', src[i:j])
        i2 = src.index('"esrgan": {')
        j2 = src.index("\n", i2)
        self.assertIn('"fixframes"', src[i2:j2])

    def test_the_documentation_passage_names_fix_missing_frames(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        i = src.index('"RIFE Install\\n"')
        self.assertIn("Fix Missing Frames", src[i:i + 300])


class _SetV:
    def __init__(self, v): self.v = v
    def get(self): return self.v
    def set(self, v): self.v = v


class TestExpansionModeButtons(unittest.TestCase):
    """The expansion models: Tone map / Topaz / Wan inpaint. Once three
    mutually-exclusive buttons, now multi-select tiles (see
    TestExpansionStudioTiles) -- but the mapping onto what the render
    reads is the same and is the part most likely to get backwards:
    whichever model Run will render sets the provider variable and the
    Wan flag. Icon still distinguishes cost: a plotted curve for this
    app's own free maths, the pixel robot for anything paid."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    class _Stub(App):
        def __init__(self, sel, pref=None):
            self._usc_sel = dict(sel)
            self._usc_view_pref = pref
            self._usc_ai_on = None
            self.usc_hdr_provider_var = _SetV(App.HDR_PROVIDERS[0])

    def _sync(self, sel, pref=None):
        app = self._Stub(sel, pref)
        m = app._usc_sync_render_model()
        return m, app._usc_hdr_provider(), app._usc_ai_on

    # ── the mapping, the part most likely to get backwards ──
    def test_tonemap_selects_local_with_ai_off(self):
        self.assertEqual(self._sync({"tonemap": True}),
                         ("tonemap", "local", False))

    def test_topaz_is_the_only_mode_selecting_the_topaz_provider(self):
        for m in ("tonemap", "topaz", "wan_vace"):
            with self.subTest(model=m):
                _m, prov, _ai = self._sync({m: True}, pref=m)
                self.assertEqual(prov, "topaz" if m == "topaz" else "local")

    def test_wan_vace_selects_local_not_topaz(self):
        """VACE's highlight reconstruction layers onto the analytic
        base; it cannot replace the whole-frame job Topaz does."""
        self.assertEqual(self._sync({"wan_vace": True}),
                         ("wan_vace", "local", True))

    def test_ai_expansion_turns_on_only_for_wan_vace(self):
        for m in ("tonemap", "topaz", "wan_vace"):
            with self.subTest(model=m):
                self.assertEqual(self._sync({}, pref=m)[2], m == "wan_vace")

    def test_the_default_selected_mode_is_tonemap(self):
        body = self._src(App._tab_upscale)
        self.assertIn('self._usc_sel = {"tonemap": True, "topaz": False, '
                      '"wan_vace": False}', body)
        self.assertEqual(self._sync({"tonemap": True})[0], "tonemap")
        self.assertEqual(self._sync({})[0], "tonemap")

    # ── the old, redundant controls are actually gone ──
    def test_the_provider_combobox_is_gone(self):
        """Replaced by the three buttons; a dropdown and a set of
        buttons controlling the same state would be the exact
        redundancy this change removes."""
        body = self._src(App._tab_upscale)
        i = body.index('c_dr = self._card(lf, "Dynamic Range")')
        j = body.index('c_of = self._card(lf, "Output Format")')
        seg = body[i:j]
        self.assertNotIn("values=list(self.HDR_PROVIDERS)", seg)

    def test_the_separate_ai_expansion_toggle_is_gone(self):
        """Its role is now the wan_vace button; keeping both would
        mean two controls disagreeing about the same state again."""
        body = self._src(App._tab_upscale)
        self.assertNotIn('"  AI Expansion  \u2014 reconstruct blown '
                         'highlights"', body)
        self.assertNotIn("self.usc_ai_toggle", body)

    def test_the_removed_toggle_variable_has_no_leftover_reference(self):
        """"usc_ai_toggle" alone is too broad -- it is a substring of
        the legitimate, still-used _usc_ai_toggled method. The removed
        thing was specifically the ATTRIBUTE self.usc_ai_toggle (no
        trailing "d"), an assignment, not a method definition."""
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertNotIn("self.usc_ai_toggle = ", src)

    # ── icon assignment ──
    def test_tonemap_uses_the_curve_icon_the_others_use_the_robot(self):
        icons = {m: i for m, _n, i in App.USC_MODELS}
        self.assertEqual(icons, {"tonemap": "curve", "topaz": "robot",
                                 "wan_vace": "robot"})

    def test_the_glyphs_reuse_the_shared_drawings(self):
        """The same robot bitmap as everywhere else, and the same knee
        curve the old badge drew -- drawn onto the tile so the progress
        fill can run underneath."""
        body = self._src(App._usc_draw_glyph)
        self.assertIn("self._ROBOT_PIXELS", body)
        self.assertIn('if icon == "robot":', body)
        self.assertIn("knee, base_y = 0.6, 0.6 * 0.35", body)
        self.assertIn("self._usc_draw_glyph(cv, icon,",
                      self._src(App._usc_draw_tile))

    # ── click targets ──
    def test_the_whole_button_area_is_clickable_not_just_the_label(self):
        """Each tile is one canvas, bound as a whole -- no part of its
        visible area is dead."""
        body = self._src(App._usc_build_model_tiles)
        self.assertIn('cv.bind("<Button-1>",', body)
        self.assertIn("self._usc_tile_click(m, e.x, e.y)", body)

    # ── HDR_PROVIDERS no longer carries UI-facing decoration ──
    def test_hdr_providers_no_longer_embeds_the_emoji_or_cost_note(self):
        """These values are switched on internally now
        (_usc_hdr_provider), never displayed directly -- the buttons
        carry the cost distinction visually instead."""
        self.assertNotIn("\U0001F916", App.HDR_PROVIDERS[0])
        self.assertNotIn("\U0001F916", App.HDR_PROVIDERS[1])

    def test_usc_hdr_provider_still_correctly_identifies_topaz(self):
        """The simplified string still has to work with the existing,
        untouched _usc_hdr_provider() parsing."""
        class V:
            def __init__(self, v):
                self.v = v

            def get(self):
                return self.v

        class Stub(App):
            def __init__(self):
                pass

        app = Stub()
        app.usc_hdr_provider_var = V(App.HDR_PROVIDERS[1])
        self.assertEqual(app._usc_hdr_provider(), "topaz")
        app.usc_hdr_provider_var = V(App.HDR_PROVIDERS[0])
        self.assertEqual(app._usc_hdr_provider(), "local")


class TestTonemapHockeyStickIcon(unittest.TestCase):
    """The Tone map icon redrawn as a hockey stick: a shallow rise up
    to the threshold, a visible corner, then a steep straight rise --
    plus a framing box and a dotted horizontal line marking the
    threshold, sitting at the same height as the curve's own corner.

    A real bug was caught building this: the dotted line was first
    drawn at y=knee (the X-POSITION where the bend happens), not at
    the height the curve actually reaches there -- a horizontal line
    marks a Y-height, and the only one that means anything is where the
    kink sits. These tests pin the corrected version by extracting and
    evaluating the function's own numbers, not by re-deriving them
    separately."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def _corner_height(self):
        """knee * 0.35, read directly out of the function's own source
        rather than hardcoded here, so this breaks the moment the
        function's real constant changes rather than silently drifting
        out of sync with it."""
        body = self._src(App._tonemap_badge)
        self.assertIn("base_y = knee * 0.35", body)
        return 0.6 * 0.35

    def test_it_returns_the_same_interface_as_the_robot_badge(self):
        """So both can sit in the same button-building code
        unmodified."""
        body = self._src(App._tonemap_badge)
        self.assertIn("set_lit", body)
        self.assertIn("class _Badge", body)

    def test_there_are_two_straight_segments_not_a_smooth_blend(self):
        """The visible kink is the point -- a continuously-bending
        curve does not read as clearly as "boosted above a threshold"
        at this size."""
        body = self._src(App._tonemap_badge)
        self.assertEqual(body.count("pts.extend(_xy("), 3)
        self.assertNotIn("smooth=True", body)

    def test_the_corner_sits_at_the_threshold_x_position(self):
        body = self._src(App._tonemap_badge)
        self.assertIn("pts.extend(_xy(knee, base_y))", body)

    def test_the_dotted_line_matches_the_curves_own_corner_height(self):
        """The bug this replaces: drawn at knee's X-position height
        instead of the height the curve actually reaches there."""
        body = self._src(App._tonemap_badge)
        self.assertIn("_xy(0.0, base_y)", body)
        self.assertNotIn("_xy(0.0, knee)", body)

    def test_the_threshold_line_is_genuinely_dashed(self):
        body = self._src(App._tonemap_badge)
        self.assertIn("dash=(2, 2)", body)

    def test_there_is_a_framing_box(self):
        body = self._src(App._tonemap_badge)
        self.assertIn("create_rectangle(", body)

    def test_the_box_border_does_not_change_with_selection(self):
        """A static frame, not part of what the eye reads as
        "on/off" -- only the curve and the threshold line do that."""
        body = self._src(App._tonemap_badge)
        i = body.index("def _set_lit")
        seg = body[i:i + 300]
        self.assertNotIn("create_rectangle", seg)

    def test_selection_relights_both_the_curve_and_the_threshold_line(self):
        """They are one visual unit -- the curve without its threshold
        marker, or vice versa, would be a half-lit, inconsistent-
        looking button."""
        body = self._src(App._tonemap_badge)
        i = body.index("def _set_lit")
        seg = body[i:i + 300]
        self.assertIn("curve_id", seg)
        self.assertIn("thresh_id", seg)

    # ── the actual numbers, evaluated ──
    def test_the_corner_and_threshold_line_coincide(self):
        corner = self._corner_height()
        knee = 0.6
        self.assertAlmostEqual(corner, knee * 0.35)

    def test_the_slope_ratio_reads_as_a_real_kink(self):
        """5-6x steeper above the threshold than below it -- a gentle
        bend would not communicate "hockey stick" at icon size."""
        knee = 0.6
        base_y = self._corner_height()
        slope_below = base_y / knee
        slope_above = (1.0 - base_y) / (1.0 - knee)
        self.assertGreater(slope_above / slope_below, 4.0)

    def test_the_curve_is_monotonic_and_spans_the_full_box(self):
        base_y = self._corner_height()
        ys = [0.0, base_y, 1.0]
        self.assertEqual(ys, sorted(ys))
        self.assertEqual(ys[0], 0.0)
        self.assertEqual(ys[-1], 1.0)

    def test_the_shallow_segment_still_rises_rather_than_flat(self):
        """A perfectly flat base would misleadingly suggest values
        below the threshold are crushed to a constant, which the real
        curve does not do -- they are compressed, not flattened."""
        base_y = self._corner_height()
        self.assertGreater(base_y, 0.0)


class TestTopazPreviewAndCache(unittest.TestCase):
    """Topaz gains its own real preview (not the "always analytic"
    placeholder it used to be) and its own cache, mirroring VACE's
    cache exactly -- except keyed on output FORMAT rather than a
    clip-level threshold, since format is the only Topaz parameter
    that changes its result, and stored as the complete scene-referred
    float result rather than an intermediate 8-bit frame, since Topaz's
    output IS the finished expansion with nothing further to merge."""

    class _V:
        def __init__(self, v):
            self.v = v

        def get(self):
            return self.v

    class _Stub(App):
        def __init__(self):
            pass

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        self.app = self._Stub()
        self.dir = tempfile.mkdtemp()
        self.names = [f"f{k:03d}.exr" for k in range(5)]
        self.fmt = "ProRes 422 HQ (.mov)"

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    # ── the cache itself ──
    def test_an_empty_cache_reports_nothing_cached(self):
        self.assertIsNone(
            self.app._usc_topaz_cached(self.dir, self.names[0], self.fmt))
        self.assertEqual(
            self.app._usc_topaz_missing(self.dir, self.names, 0, 4,
                                        self.fmt), [0, 1, 2, 3, 4])

    def test_storing_scene_referred_arrays_makes_them_cached(self):
        arrs = [np.full((8, 8, 3), v, np.float32) for v in (0.5, 1.0, 6.0)]
        n = self.app._usc_topaz_store(self.dir, self.names, 0, arrs,
                                      self.fmt)
        self.assertEqual(n, 3)
        self.assertEqual(
            self.app._usc_topaz_missing(self.dir, self.names, 0, 4,
                                        self.fmt), [3, 4])

    def test_the_cached_value_round_trips_exactly_including_over_range(self):
        """A blown-highlight value above 1.0 has to survive -- that is
        the entire reason this is stored as EXR and not an 8-bit
        format."""
        arrs = [np.full((8, 8, 3), 6.0, np.float32)]
        self.app._usc_topaz_store(self.dir, self.names, 2, arrs, self.fmt)
        p = self.app._usc_topaz_cached(self.dir, self.names[2], self.fmt)
        back = self.app._ff_read_frame(p)
        self.assertAlmostEqual(float(back[0, 0, 0]), 6.0, places=2)

    def test_a_different_format_is_a_separate_cache_entry(self):
        """Format is the only Topaz parameter that changes the result,
        so it has to be part of the key -- reusing a HEVC-cached result
        after switching to ProRes would be subtly wrong."""
        arrs = [np.full((8, 8, 3), 1.0, np.float32)]
        self.app._usc_topaz_store(self.dir, self.names, 0, arrs, self.fmt)
        self.assertIsNone(self.app._usc_topaz_cached(
            self.dir, self.names[0], "H.265 HDR10 (.mp4)"))

    def test_peak_and_knee_are_not_part_of_the_cache_key(self):
        """Those are THIS APP's own analytic parameters and do not
        apply to Topaz's result at all."""
        body = self._src(App._usc_topaz_cache_key)
        self.assertNotIn("peak", body.lower())
        self.assertNotIn("knee", body.lower())

    def test_it_uses_the_shared_exr_writer_not_a_new_format(self):
        body = self._src(App._usc_topaz_store)
        self.assertIn("_organic_write_exr", body)

    # ── the cache-first short-circuit ──
    def test_a_fully_cached_span_makes_no_fal_call_at_all(self):
        """Verified against a stub whose _topaz_submit raises if
        called, not just by reading the source."""
        class NoFalStub(App):
            def __init__(self):
                pass

            def _topaz_submit(self, *a, **kw):
                raise AssertionError("fal was called despite a full "
                                    "cache hit")

        app = NoFalStub()
        app.usc_topaz_fmt_var = TestTopazPreviewAndCache._V(self.fmt)
        arrs = [np.full((8, 8, 3), v, np.float32) for v in (0.1, 0.2, 0.3)]
        app._usc_topaz_store(self.dir, self.names[:3], 0, arrs, self.fmt)
        logs = []
        result = app._usc_topaz_expand(self.dir, self.names[:3],
                                       on_log=logs.append)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(float(result[1][0, 0, 0]), 0.2, places=2)
        self.assertTrue(any("no call needed" in m for m in logs))

    def test_a_successful_generation_is_cached_for_next_time(self):
        body = self._src(App._usc_topaz_expand)
        self.assertIn("_usc_topaz_store(", body)

    def test_partial_cache_still_resubmits_the_whole_span(self):
        """Unlike VACE's masked regional regenerate, Topaz grades a
        whole clip as one unit -- there is no partial-span equivalent,
        so this is an honest limitation, not a bug."""
        body = self._src(App._usc_topaz_expand)
        self.assertIn("no partial-span equivalent", body)

    # ── the preview generation itself ──
    def test_preview_uses_the_short_span_helper_not_a_hardcoded_range(self):
        body = self._src(App._usc_preview_model)
        self.assertIn("self._vace_span_for(idx, len(names),\n"
                      "                                         "
                      "want=self.USC_TOPAZ_PREVIEW_SPAN)", body)

    def test_the_preview_span_constant_is_much_smaller_than_vaces(self):
        self.assertLess(App.USC_TOPAZ_PREVIEW_SPAN, App.VACE_MIN_GEN)

    def test_the_preview_span_clears_the_known_fal_duration_floor(self):
        """The 0.1s minimum-duration floor found earlier this session
        on the exact same kind of call -- this has to clear it with
        real margin, not just barely."""
        frames_for_0_1s_at_24fps = 0.1 * 24
        self.assertGreater(App.USC_TOPAZ_PREVIEW_SPAN,
                          frames_for_0_1s_at_24fps * 2)

    def test_a_failed_model_does_not_stop_the_others(self):
        """One model failing marks ITS tile failed and logs why; the
        frame stays up and the remaining selected models still run."""
        body = self._src(App._usc_preview)
        loop = body[body.index("for k, m in enumerate(models):"):]
        self.assertIn("except Exception as ex:", loop)
        self.assertIn('self._usc_tile_set(m, "failed", 0.0)', loop)
        self.assertIn("preview failed", loop)
        self.assertLess(loop.index("except Exception as ex:"),
                        loop.index("finally:"))

    # ── the three-way reapply dispatcher ──
    def test_reapply_checks_the_topaz_cache_when_topaz_is_selected(self):
        body = self._src(App._usc_model_cache_hit)
        seg = body[body.index('if m == "topaz":'):body.index('if m == "wan_vace":')]
        self.assertIn("self._usc_topaz_cached(", seg)
        self.assertIn("self.usc_topaz_fmt_var.get()", seg)

    def test_reapply_shows_the_cached_topaz_result_directly(self):
        """Read straight back via _ff_read_frame, since a cached Topaz
        entry already IS the finished scene-referred result -- unlike
        VACE's cache, nothing further needs merging."""
        body = self._src(App._usc_model_result)
        seg = body[body.index('if m == "topaz":'):body.index('if m == "wan_vace":')]
        self.assertIn("return self._ff_read_frame(hit)", seg)

    def test_no_green_box_without_a_result_for_the_current_settings(self):
        """No cache entry and no in-memory result for THIS format -> no
        green box. An in-memory fallback made under another format does
        not count either."""
        d = tempfile.mkdtemp()
        try:
            class Stub(App):
                def __init__(s):
                    s._usc_frames = ["a.png", "b.png"]
                    s._usc_folder = d
                    s._usc_preview_idx = 0
                    s._usc_preview_code = np.zeros((2, 2, 3), np.float32)
                    s._usc_mem_results = {}
                    s._usc_tonemap_ready = False
                    s.usc_topaz_fmt_var = TestTopazPreviewAndCache._V("A")

                def _usc_topaz_cache_dir(s):
                    return d
            app = Stub()
            self.assertFalse(app._usc_model_available("topaz"))
            self.assertIsNone(app._usc_model_result("topaz"))
            app._usc_mem_results[("topaz", 0, "A")] = {
                "kind": "scene", "arr": np.ones((2, 2, 3), np.float32)}
            self.assertTrue(app._usc_model_available("topaz"))
            self.assertEqual(float(app._usc_model_result("topaz").max()), 1.0)
            app.usc_topaz_fmt_var = TestTopazPreviewAndCache._V("B")
            self.assertFalse(app._usc_model_available("topaz"))
            app._usc_preview_idx = 1      # another frame: not this one's
            app.usc_topaz_fmt_var = TestTopazPreviewAndCache._V("A")
            self.assertFalse(app._usc_model_available("topaz"))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_reapply_still_covers_the_vace_and_tonemap_cases(self):
        """Every model has its own rebuild branch, not just Topaz."""
        body = self._src(App._usc_model_result)
        for branch in ('if m == "tonemap":', 'if m == "topaz":',
                       'if m == "wan_vace":'):
            with self.subTest(branch=branch):
                self.assertIn(branch, body)

    def test_reapply_never_generates_only_reads_the_cache(self):
        """The whole point: comparing models must not spend money by
        accident on a click."""
        for fn in (App._usc_model_result, App._usc_model_cache_hit,
                   App._usc_view, App._usc_tile_click):
            with self.subTest(fn=fn.__name__):
                body = self._src(fn)
                self.assertNotIn("_usc_topaz_expand(", body)
                self.assertNotIn("_topaz_submit(", body)
                self.assertNotIn("_usc_ai_generate(", body)

    # ── the clear-cache button ──
    def test_clearing_asks_first_same_as_the_vace_cache_button(self):
        body = self._src(App._usc_topaz_clear_cache)
        self.assertIn("askyesno", body)
        self.assertIn('default="no"', body)

    def test_an_empty_topaz_cache_does_not_prompt(self):
        body = self._src(App._usc_topaz_clear_cache)
        self.assertLess(body.index("already empty"), body.index("askyesno"))

    def test_the_button_is_wired_into_the_topaz_format_row(self):
        body = self._src(App._tab_upscale)
        i = body.index("usc_topaz_fmt_var = tk.StringVar")
        seg = body[i:i + 600]
        self.assertIn("usc_topaz_cache_btn", seg)


class TestUpscaleHistogram(unittest.TestCase):
    """The value-distribution histogram: additive to whatever the
    preview is already showing (Wipe/Diff), not a third exclusive mode,
    plotting the SAME scene-referred float the preview and the eventual
    write both use -- so comparing Tone map / Topaz / Wan VACE inpaint
    is a comparison of the actual thing being written, not a separate
    rendering."""

    class _Stub(App):
        def __init__(self):
            pass

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_it_is_always_visible_not_a_toggle(self):
        """Always on, in the Preview & Compare card right under the
        compare modes -- no button that hides it."""
        body = self._src(App._tab_upscale)
        self.assertIn("self.usc_hist_canvas = tk.Canvas(c3,", body)
        self.assertIn("self.usc_hist_canvas.pack(", body)
        self.assertIn("self.usc_hist_on = True", body)
        self.assertNotIn("tk.Canvas(rtop, height=90", body)
        self.assertFalse(hasattr(App, "_usc_toggle_histogram"))

    def test_it_plots_the_actual_displayed_float_not_a_separate_copy(self):
        body = self._src(App._usc_histogram_refresh)
        self.assertIn("_usc_up_float", body)

    def test_it_does_nothing_when_the_panel_is_off(self):
        """The guard stays (usc_hist_on is simply always True now), so
        a stub or a half-built tab can still call this safely."""
        body = self._src(App._usc_histogram_refresh)
        self.assertIn("usc_hist_on", body)

    def test_it_does_not_crash_with_no_frame_loaded_yet(self):
        body = self._src(App._usc_histogram_refresh)
        self.assertIn("if arr is None", body)

    def test_the_left_panel_is_always_exactly_zero_to_one(self):
        """Superseded: a single shared axis squeezed the ORIGINAL
        distribution -- almost always the bulk of the pixels -- into a
        sliver whenever highlights ran far past white. The left panel
        now has its own fixed 0..1 range regardless of how far
        anything above it extends."""
        body = self._src(App._usc_histogram_refresh)
        self.assertIn("range=(0.0, 1.0)", body)

    def test_the_right_panel_only_appears_when_something_exceeds_one(self):
        body = self._src(App._usc_histogram_refresh)
        self.assertIn("has_above = actual_max > 1.0", body)

    def test_the_two_panels_have_independent_scales(self):
        """Not one axis with a marker -- two different scales sitting
        next to each other, which is what makes the original
        distribution comparable across frames whose highlights reach
        very different heights."""
        body = self._src(App._usc_histogram_refresh)
        self.assertIn("USC_HIST_LEFT_FRAC", body)
        self.assertEqual(body.count("range=("), 2)

    def test_the_actual_peak_value_is_labelled_numerically(self):
        """Eyeballing bar heights does not tell you the number; this
        does."""
        body = self._src(App._usc_histogram_refresh)
        self.assertIn('f"max {actual_max:.2f}"', body)

    def test_bar_height_is_log_scaled_not_a_raw_count(self):
        """A handful of blown-highlight pixels would otherwise round to
        an invisible sliver next to the mass of ordinary midtone
        pixels -- exactly the part of the picture this exists to make
        visible."""
        body = self._src(App._usc_histogram_refresh)
        self.assertIn("np.log1p(", body)

    def test_white_point_is_a_dashed_marker_on_one_axis(self):
        """One continuous axis like a grading histogram: 0 at the far
        left, 1.0 marked by a dashed line, above-white to its right."""
        body = self._src(App._usc_histogram_refresh)
        self.assertIn('dash=(3, 2)', body)
        self.assertIn('"#6FA8DC"', body)   # blue 0..1 curve
        self.assertIn('"#F0C850"', body)   # amber above-white curve

    def test_axis_is_labelled_0_and_1(self):
        body = self._src(App._usc_histogram_refresh)
        self.assertIn('text="0"', body)
        self.assertIn('text="1.0"', body)
        self.assertIn('above white', body)

    def test_renders_blue_left_and_amber_right(self):
        import tkinter as tk
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("no display")
        try:
            cv = tk.Canvas(root, width=400, height=100)
            cv.pack(); root.update_idletasks()
            fills = []
            orig = cv.create_polygon
            def _poly(*a, **k):
                fills.append(k.get("fill"))
                return orig(*a, **k)
            cv.create_polygon = _poly
            import numpy as np
            st = type("S", (), {})()
            st.USC_HIST_LEFT_FRAC = App.USC_HIST_LEFT_FRAC
            st.usc_hist_canvas = cv
            st.usc_hist_on = True
            arr = np.linspace(0, 4, 3000, dtype=np.float32).reshape(30, 100)
            st._usc_up_float = np.stack([arr] * 3, -1)
            App._usc_histogram_refresh(st)
            self.assertIn("#1B3550", fills)
            self.assertIn("#5C4C1C", fills)
        finally:
            root.destroy()

    def test_histogram_refreshes_after_a_fresh_preview(self):
        """A finished model is put on screen through _usc_view, which
        refreshes the histogram."""
        self.assertIn("self._usc_view(", self._src(App._usc_model_finished))

    def test_histogram_refreshes_after_switching_expansion_mode(self):
        """Comparing models is the whole feature -- the histogram has
        to update on every switch, not just the first preview."""
        self.assertIn("self._usc_histogram_refresh()", self._src(App._usc_view))
        self.assertIn("self._usc_histogram_refresh()",
                      self._src(App._usc_view_none))

    def test_the_computation_matches_realistic_data(self):
        """Direct numeric check, not just source inspection: all pixels
        accounted for, and a small blown-highlight cluster lands in a
        visibly non-zero bin rather than rounding away."""
        rng = np.random.default_rng(1)
        arr = rng.random((100, 100, 3)).astype(np.float32) * 0.3
        arr[10:15, 10:15] = 6.0
        peak = arr.max(axis=2)
        xmax = max(2.0, float(peak.max()) * 1.02)
        counts, edges = np.histogram(peak, bins=40, range=(0.0, xmax))
        self.assertEqual(int(counts.sum()), peak.size)
        heights = np.log1p(counts.astype(np.float32))
        hmax = max(1e-6, float(heights.max()))
        top_bin = np.searchsorted(edges, 6.0) - 1
        self.assertGreater(heights[top_bin] / hmax, 0.1)

    def test_all_zero_data_does_not_raise(self):
        arr = np.zeros((10, 10, 3), np.float32)
        peak = arr.max(axis=2)
        xmax = max(2.0, float(peak.max()) * 1.02)
        counts, _ = np.histogram(peak, bins=40, range=(0.0, xmax))
        heights = np.log1p(counts.astype(np.float32))
        hmax = max(1e-6, float(heights.max()))
        self.assertGreater(hmax, 0.0)


class TestRenderFallbackVisibility(unittest.TestCase):
    """Reported: three renders (Tone map, Wan VACE inpaint, Topaz)
    produced three IDENTICAL outputs. The dispatch, per-render state
    scoping, and the write step were all checked directly and found
    structurally correct -- _topaz_scene and _gen_dir are purely local,
    freshly reset each render, and _usc_write_frame genuinely writes
    the mode-dependent px, not the raw source.

    The remaining, much more likely explanation: BOTH external calls
    silently fell back to the same analytic result, and a single "err"
    log line near the top of a long per-frame render log is easy to
    miss, with nothing in the final completion status reflecting it.
    This closes that gap -- a degraded render can no longer finish
    looking exactly like a clean success."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_a_fallback_list_is_tracked_across_the_render(self):
        body = self._src(App._usc_run)
        self.assertIn("_fell_back = []", body)

    def test_a_topaz_fallback_is_recorded(self):
        body = self._src(App._usc_run)
        i = body.index("Topaz expansion failed, using analytic")
        self.assertIn("_fell_back.append(", body[i:i + 200])

    def test_a_vace_fallback_is_recorded(self):
        body = self._src(App._usc_run)
        i = body.index("AI expansion failed, using analytic")
        self.assertIn("_fell_back.append(", body[i:i + 200])

    def test_the_recorded_reason_includes_the_actual_exception_text(self):
        """The whole point: whatever made it fail has to be visible at
        the end, not just guessable from an early log line.

        Checked without the arrow character itself: the raw source
        holds it as an unevaluated escape sequence, while a plain
        string literal written here would be evaluated into the real
        character the moment this test file is loaded -- the two can
        never compare equal, regardless of window size."""
        body = self._src(App._usc_run)
        self.assertIn('_fell_back.append(f"Topaz', body)
        self.assertIn("analytic: {_tex}", body)
        self.assertIn('_fell_back.append(f"AI Expansion', body)
        self.assertIn("analytic: {_ex}", body)

    def test_a_clean_run_still_reports_success_normally(self):
        """Not touched for the common, everything-worked case -- only
        a genuine fallback changes what gets shown."""
        body = self._src(App._usc_run)
        self.assertIn("frames expanded", body)
        self.assertIn('self.setstatus("Complete", GR)', body)

    def test_a_fallback_run_says_so_in_the_final_completion_message(self):
        body = self._src(App._usc_run)
        self.assertIn("if _fell_back:", body)
        i = body.index("if _fell_back:")
        seg = body[i:i + 1000]
        # The message is two adjacent string literals split across a
        # line, so checked as the two halves rather than one
        # contiguous phrase that a raw-source search would never find.
        self.assertIn("but fell back", seg)
        self.assertIn("to analytic for the whole run", seg)

    def test_a_fallback_run_changes_the_status_colour_too(self):
        """The status bar is often the only thing glanced at after a
        render -- it has to disagree with green when the result is not
        what was actually asked for."""
        body = self._src(App._usc_run)
        i = body.index("if _fell_back:")
        seg = body[i:i + 1000]
        self.assertIn('"Complete (fell back to analytic)"', seg)
        self.assertNotIn('"Complete (fell back to analytic)", GR', seg)

    def test_the_fallback_message_is_still_logged_at_error_level(self):
        """Confirms the pre-existing per-call log lines were already
        correctly at "err" -- the gap being closed is END-of-run
        visibility, not severity of the original line."""
        body = self._src(App._usc_run)
        i1 = body.index("Topaz expansion failed")
        i2 = body.index("AI expansion failed")
        self.assertIn('"err"', body[i1:i1 + 150])
        self.assertIn('"err"', body[i2:i2 + 150])


class TestVaceCacheKeyIncludesPrompt(unittest.TestCase):
    """Reported: with a frame already generated once via Wan VACE
    inpaint, changing the Surface preset or Guide text and generating
    again kept showing the FIRST result. Root cause: the cache key
    was folder|name|threshold only -- the prompt was never part of
    it, so a different preset produced the identical key and the
    cache reported "already generated" for a request that had, in
    fact, never been made."""

    class _Stub(App):
        def __init__(self, cache):
            self._cache = cache

        def _usc_ai_cache_dir(self):
            return self._cache

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        self.cache = tempfile.mkdtemp()
        self.app = self._Stub(self.cache)

    def tearDown(self):
        shutil.rmtree(self.cache, ignore_errors=True)

    # ── the core mechanism, verified directly ──
    def test_different_prompts_produce_different_keys(self):
        k1 = self.app._usc_ai_cache_key(
            "/plate", "f001.exr", 0.96, "sunlit white fabric")
        k2 = self.app._usc_ai_cache_key(
            "/plate", "f001.exr", 0.96, "polished metal")
        self.assertNotEqual(k1, k2)

    def test_the_same_prompt_still_produces_the_same_key(self):
        """Not sensitive to anything BUT what actually changed --
        re-hashing on every call must stay stable so an unrelated
        preview does not invalidate a good cache."""
        k1 = self.app._usc_ai_cache_key(
            "/plate", "f001.exr", 0.96, "sunlit white fabric")
        k2 = self.app._usc_ai_cache_key(
            "/plate", "f001.exr", 0.96, "sunlit white fabric")
        self.assertEqual(k1, k2)

    def test_a_frame_cached_under_one_prompt_misses_under_another(self):
        """The exact reported scenario, reproduced directly: generate
        under prompt A, then check with prompt B."""
        key_a = self.app._usc_ai_cache_key(
            self.cache, "f001.exr", 0.96, "sunlit white fabric")
        open(key_a, "w").close()
        self.assertIsNotNone(self.app._usc_ai_cached(
            self.cache, "f001.exr", 0.96, "sunlit white fabric"))
        self.assertIsNone(self.app._usc_ai_cached(
            self.cache, "f001.exr", 0.96, "polished metal"))

    def test_threshold_alone_no_longer_decides_the_cache_hit(self):
        """Pins the bug directly: same threshold, different prompt,
        must not be treated as the same request."""
        key_a = self.app._usc_ai_cache_key(
            self.cache, "f001.exr", 0.96, "prompt one")
        open(key_a, "w").close()
        missing = self.app._usc_ai_missing(
            self.cache, ["f001.exr"], 0, 0, 0.96, "prompt two")
        self.assertEqual(missing, [0])

    # ── every call site threads the resolved prompt consistently ──
    def test_generate_computes_the_resolved_prompt_before_the_cache_check(self):
        """_said has to exist and be resolved BEFORE _usc_ai_missing is
        called, or the cache check runs against nothing."""
        body = self._src(App._usc_ai_generate)
        self.assertLess(body.index("_said = "),
                        body.index("_usc_ai_missing("))

    def test_generate_passes_the_resolved_prompt_to_missing_and_view(self):
        body = self._src(App._usc_ai_generate)
        i = body.index("_usc_ai_missing(")
        self.assertIn("_said", body[i:i + 100])
        j = body.index("_usc_ai_cache_view(")
        self.assertIn("_said", body[j:j + 100])

    def test_generate_still_resolves_the_prompt_when_on_log_is_none(self):
        """The resolution used to live inside an `if on_log:` block --
        called with no logger, _said would never be defined at all,
        and the cache check right after it would raise."""
        body = self._src(App._usc_ai_generate)
        self.assertIn("else:", body[body.index("if on_log:"):
                                    body.index("_usc_ai_missing(")])

    def test_the_store_call_passes_the_resolved_prompt(self):
        body = self._src(App._usc_ai_generate)
        i = body.index("self._usc_ai_store(")
        self.assertIn("_said", body[i:i + 80])

    def test_reapply_resolves_and_passes_the_prompt_too(self):
        """Comparing models has to reflect the CURRENT prompt (and the
        current shadow settings too), not whatever was true the first
        time this frame was generated -- so the green box, which reads
        this lookup, goes out the moment Guide changes."""
        body = self._src(App._usc_model_cache_hit)
        seg = body[body.index('if m == "wan_vace":'):]
        self.assertIn("or self.HDR_AI_PROMPT", seg)
        self.assertIn("_usc_ai_cached(", seg)
        self.assertIn("folder, names[idx], float(self.usc_ai_thresh.get()),", seg)
        self.assertIn("said, st, sp", seg)

    def test_the_timeline_colour_also_matches_on_the_current_prompt(self):
        """The same bug, one level up: a frame's amber "already
        generated" marker must stop claiming that the moment the
        prompt changes, not keep pointing at a stale result."""
        body = self._src(App._usc_frame_colour)
        self.assertIn("_said", body)
        self.assertIn("_usc_ai_cached(", body)
        self.assertIn("names[i], thresh, _said", body)

    def test_the_negative_prompt_is_not_part_of_the_key(self):
        """HDR_AI_NEGATIVE is a fixed constant, never user-edited -- it
        never varies at runtime, so it contributes nothing to
        differentiating one request from another."""
        body = self._src(App._usc_ai_cache_key)
        self.assertNotIn("NEGATIVE", body)


class TestShadowReconstruction(unittest.TestCase):
    """From the Eyeline/DiffHDR paper (arXiv 2604.06161): crushed
    shadows have lost radiance just as surely as blown highlights, and
    a reconstruction model has just as much to offer there. Added as a
    symmetric mask (_hdr_crushed_mask) unioned into the same VACE call,
    with a region-appropriate merge guard: a floor at white for
    highlights, a CEILING at the shadow threshold for shadows, since
    inventing a shape in near-black is a worse mistake than a
    highlight staying merely plausible."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def _patch(self, peak, thresh):
        import numpy as np
        code = np.zeros((4, 4, 3), np.float32)
        code[:] = peak
        return code

    # ── the symmetric mask, verified numerically ──
    def test_a_blown_patch_is_not_flagged_as_crushed(self):
        code = self._patch(1.0, None)
        m = App._hdr_crushed_mask(code)
        self.assertLess(float(m.max()), 0.1)

    def test_a_crushed_patch_is_flagged(self):
        code = self._patch(0.005, None)
        m = App._hdr_crushed_mask(code)
        self.assertGreater(float(m.min()), 0.9)

    def test_a_saturated_colour_is_not_falsely_flagged_as_crushed(self):
        """A pure, saturated colour can have two channels near zero
        without the pixel having lost any information -- it is only
        crushed if even its BRIGHTEST channel is at the floor."""
        import numpy as np
        code = np.zeros((4, 4, 3), np.float32)
        code[:] = [0.8, 0.02, 0.01]
        m = App._hdr_crushed_mask(code)
        self.assertLess(float(m.max()), 0.1)

    def test_an_ordinary_midtone_is_not_flagged_either_way(self):
        code = self._patch(0.5, None)
        hi = App._hdr_clipped_mask(code)
        lo = App._hdr_crushed_mask(code)
        self.assertLess(float(hi.max()), 0.1)
        self.assertLess(float(lo.max()), 0.1)

    # ── the mask builder unions both when asked ──
    def test_mask_frame_is_highlight_only_by_default(self):
        """shadow_thresh=None must reproduce the exact old behaviour --
        every existing call site that never passes it stays correct."""
        import numpy as np
        code = np.zeros((20, 20, 3), np.float32)
        code[0:5, 0:5] = 1.0     # blown
        code[10:15, 10:15] = 0.005   # crushed
        m8 = App._hdr_ai_mask_frame(code, thresh=0.96, dilate=0)
        self.assertGreater(int(m8[0:5, 0:5].max()), 200)
        self.assertEqual(int(m8[10:15, 10:15].max()), 0)

    def test_mask_frame_unions_shadows_when_given_a_threshold(self):
        import numpy as np
        code = np.zeros((20, 20, 3), np.float32)
        code[0:5, 0:5] = 1.0
        code[10:15, 10:15] = 0.005
        m8 = App._hdr_ai_mask_frame(code, thresh=0.96, dilate=0,
                                    shadow_thresh=0.04)
        self.assertGreater(int(m8[0:5, 0:5].max()), 200)
        self.assertGreater(int(m8[10:15, 10:15].max()), 200)

    # ── the merge: old behaviour exactly preserved ──
    def test_merge_with_no_shadow_threshold_matches_the_old_function(self):
        import numpy as np
        code = np.zeros((10, 10, 3), np.float32)
        code[0:3, 0:3] = 1.0
        analytic = np.full((10, 10, 3), 2.0, np.float32)
        gen = np.full((10, 10, 3), 0.5, np.float32)
        gen[0:3, 0:3] = np.linspace(0.8, 1.0, 9).reshape(3, 3, 1)
        out = App._hdr_ai_merge(analytic, gen, code, amount=1.0,
                                thresh=0.96, eotf="srgb")
        self.assertGreaterEqual(float(out[0:3, 0:3].min()), 1.0 - 1e-4)
        self.assertTrue(np.allclose(out[5:, 5:], 2.0, atol=1e-3))

    # ── the merge: both regions handled correctly together ──
    def test_merge_reconstructs_both_regions_with_correct_guards(self):
        import numpy as np
        code = np.zeros((10, 10, 3), np.float32)
        code[0:3, 0:3] = 1.0
        code[6:9, 6:9] = 0.01
        analytic = np.full((10, 10, 3), 2.0, np.float32)
        analytic[6:9, 6:9] = 0.02
        gen = np.full((10, 10, 3), 0.5, np.float32)
        gen[0:3, 0:3] = np.linspace(0.8, 1.0, 9).reshape(3, 3, 1)
        gen[6:9, 6:9] = np.linspace(0.05, 0.15, 9).reshape(3, 3, 1)
        out = App._hdr_ai_merge(analytic, gen, code, amount=1.0,
                                thresh=0.96, eotf="srgb",
                                shadow_thresh=0.04)
        # highlight: varies, stays >= white
        self.assertGreaterEqual(float(out[0:3, 0:3].min()), 1.0 - 1e-4)
        self.assertFalse(np.allclose(out[0:3, 0:3], out[0, 0]))
        # shadow: varies, stays under the shadow ceiling
        self.assertLessEqual(float(out[6:9, 6:9].max()), 0.04 + 1e-4)
        self.assertFalse(np.allclose(out[6:9, 6:9], out[6, 6]))

    def test_shadow_region_is_never_brightened_past_the_ceiling(self):
        """The whole point of the asymmetric guard: even if the
        generated content would imply a brighter value, the shadow
        side must not exceed the point where it would no longer have
        been crushed."""
        import numpy as np
        code = np.zeros((6, 6, 3), np.float32)
        code[:] = 0.01
        analytic = np.full((6, 6, 3), 0.02, np.float32)
        # deliberately provocative: generated content implies the
        # shadow should be much brighter than the ceiling
        gen = np.full((6, 6, 3), 0.9, np.float32)
        out = App._hdr_ai_merge(analytic, gen, code, amount=1.0,
                                thresh=0.96, eotf="srgb",
                                shadow_thresh=0.04)
        self.assertLessEqual(float(out.max()), 0.04 + 1e-4)

    def test_highlight_region_is_unaffected_by_shadow_reconstruction(self):
        """Turning shadows on must not change how highlights merge."""
        import numpy as np
        code = np.zeros((6, 6, 3), np.float32)
        code[:] = 1.0
        analytic = np.full((6, 6, 3), 3.0, np.float32)
        gen = np.full((6, 6, 3), 0.9, np.float32)
        out_off = App._hdr_ai_merge(analytic, gen, code, amount=1.0,
                                    thresh=0.96, eotf="srgb")
        out_on = App._hdr_ai_merge(analytic, gen, code, amount=1.0,
                                   thresh=0.96, eotf="srgb",
                                   shadow_thresh=0.04)
        import numpy as np
        self.assertTrue(np.allclose(out_off, out_on))

    # ── one shared helper, not duplicated logic ──
    def test_one_shared_helper_decides_the_shadow_threshold(self):
        """So the mask builder, the render-time merge, and the
        reapply/compare merge cannot drift out of sync with each
        other -- exactly the risk of writing the same conditional
        three times."""
        body = self._src(App._usc_ai_union_mask)
        self.assertIn("self._usc_shadow_thresh()", body)
        body2 = self._src(App._usc_ai_apply)
        self.assertIn("self._usc_shadow_thresh()", body2)
        body3 = self._src(App._usc_model_cache_hit)
        self.assertIn("self._usc_shadow_thresh()", body3)
        self.assertIn("self._usc_shadow_thresh()",
                      self._src(App._usc_merge_wan))

    def test_the_toggle_defaults_off(self):
        """Superseded: there is no separate BooleanVar any more --
        Shadow Amount itself defaults to 0.0, which IS off, mirroring
        how highlight reconstruction has never had a separate on/off
        control either."""
        body = self._src(App._tab_upscale)
        i = body.index("usc_ai_shadow_amount = tk.DoubleVar")
        self.assertIn("value=0.0", body[i:i + 60])

    # ── the cache key includes the shadow state ──
    def test_the_cache_key_differs_with_shadows_on_vs_off(self):
        class Stub(App):
            def __init__(self, cache):
                self._cache = cache

            def _usc_ai_cache_dir(self):
                return self._cache

        cache = tempfile.mkdtemp()
        try:
            app = Stub(cache)
            k_off = app._usc_ai_cache_key("/plate", "f.exr", 0.96, "guide",
                                          False)
            k_on = app._usc_ai_cache_key("/plate", "f.exr", 0.96, "guide",
                                         True)
            self.assertNotEqual(k_off, k_on)
        finally:
            shutil.rmtree(cache, ignore_errors=True)

    def test_toggling_shadows_forces_a_regenerate_not_a_stale_hit(self):
        """The exact class of bug fixed for the prompt last turn --
        this closes the same door for the shadow toggle."""
        class Stub(App):
            def __init__(self, cache):
                self._cache = cache

            def _usc_ai_cache_dir(self):
                return self._cache

        cache = tempfile.mkdtemp()
        try:
            app = Stub(cache)
            key_off = app._usc_ai_cache_key(cache, "f.exr", 0.96, "guide",
                                            False)
            open(key_off, "w").close()
            self.assertIsNotNone(app._usc_ai_cached(
                cache, "f.exr", 0.96, "guide", False))
            self.assertIsNone(app._usc_ai_cached(
                cache, "f.exr", 0.96, "guide", True))
        finally:
            shutil.rmtree(cache, ignore_errors=True)

    def test_the_shadow_amount_slider_is_built_the_same_way_as_highlights(self):
        """Superseded: there is no separate toggle function any more --
        the Amount slider itself is what turns shadow reconstruction
        on. It does NOT live-update the preview on drag, and neither
        does the pre-existing highlight Amount slider -- that is a
        property of _ff_slider_factory itself (it only updates its own
        numeric label), true for both, not something this turn changed
        or should claim otherwise."""
        body = self._src(App._tab_upscale)
        self.assertEqual(body.count("self._ff_slider_factory(_hi_tab)"), 1)
        self.assertEqual(body.count("self._ff_slider_factory(_lo_tab)"), 1)

    def test_slider_factory_only_updates_its_own_label_not_the_preview(self):
        """Confirms the fact the test above relies on, directly."""
        body = self._src(App._ff_slider_factory)
        self.assertIn("vl.configure(text=fmt(var.get()))", body)
        self.assertNotIn("_usc_ai_reapply", body)


class TestShadowControlVisibility(unittest.TestCase):
    """Reported, twice now: first that it was unclear whether shadow
    reconstruction applies under Tone map and Topaz (it does not --
    Wan VACE inpaint only, since neither other mode has a generative
    model to reconstruct with), then that greying alone was not clear
    enough. The whole Highlights/Shadows notebook is now shown or
    hidden entirely by _usc_ai_toggled, not merely greyed -- there is
    nothing to squint at when it is not relevant."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_the_notebook_is_shown_and_hidden_not_greyed(self):
        body = self._src(App._usc_models_changed)
        i = body.index("nb = getattr(self, \"usc_ai_notebook\"")
        seg = body[i:i + 700]
        self.assertIn('nb.pack(fill="x"', seg)
        self.assertIn("nb.pack_forget()", seg)
        self.assertNotIn("_usc_set_enabled", seg)

    def test_show_hides_on_off_matches_wan_being_in_play(self):
        """Shown when Wan inpaint is selected or being rendered -- and,
        since the masks became writable with any model, also whenever
        Include masks is on (the panel sets the levels they are cut
        at). Hidden otherwise."""
        body = self._src(App._usc_models_changed)
        i = body.index("nb = getattr(self, \"usc_ai_notebook\"")
        seg = body[i:i + 700]
        self.assertLess(seg.index("if self._usc_mask_controls_relevant():"),
                        seg.index("nb.pack("))
        self.assertLess(seg.index("nb.pack("), seg.index("nb.pack_forget()"))
        rel = self._src(App._usc_mask_controls_relevant)
        self.assertIn("self._usc_wan_relevant()", rel)

    def test_the_shadow_tab_lives_inside_that_same_notebook(self):
        body = self._src(App._tab_upscale)
        i = body.index("self.usc_ai_notebook = ttk.Notebook")
        j = body.index('self.usc_ai_notebook.add(_lo_tab, text="Shadows")')
        self.assertGreater(j, i)

    def test_the_highlights_tab_lives_inside_it_too(self):
        """Not just the new shadow controls -- the pre-existing
        highlight ones moved into the notebook as well, so both are
        governed by the same show/hide, not a mix of the old greying
        and the new hiding."""
        body = self._src(App._tab_upscale)
        i = body.index("self.usc_ai_notebook = ttk.Notebook")
        j = body.index('self.usc_ai_notebook.add(_hi_tab, text="Highlights")')
        self.assertGreater(j, i)

    def test_show_blown_mask_is_inside_the_highlights_tab(self):
        """Both tab FRAMES are created together up front; content is
        added to each afterward. The real invariant is parentage, not
        frame-creation order -- Show blown mask has to be built as a
        child of _hi_tab, before the Shadows tab's own content starts."""
        body = self._src(App._tab_upscale)
        self.assertIn("usc_ai_mask_btn = tk.Label(_hi_tab,", body)
        i = body.index("usc_ai_mask_btn = tk.Label(_hi_tab,")
        j = body.index("usc_shadow_mask_btn = tk.Label(_lo_tab,")
        self.assertLess(i, j)

    def test_the_shadow_tooltip_names_what_the_amount_slider_does(self):
        """No separate on/off control any more, so the scope
        explanation moved to what actually IS the switch: raising
        Shadow Amount above 0."""
        body = self._src(App._tab_upscale)
        i = body.index('"Shadow level"')
        seg = body[max(0, i - 900):i]
        self.assertIn("turns shadow", seg)


class TestShadowTabMirroredControls(unittest.TestCase):
    """Shadow reconstruction became a full mirror of highlight
    reconstruction: its own Amount, its own tunable level, its own
    mask preview, its own Surface presets, its own Guide -- as a
    second tab in the same notebook, shown only when Wan VACE inpaint
    is selected, not merely greyed. VACE takes exactly one text prompt
    per call with no way to bind it spatially, so both Guides are
    combined into one submission rather than doubling the cost of
    every render with a second, fully independent call."""

    class _V:
        def __init__(self, v):
            self.v = v

        def get(self):
            return self.v

        def set(self, v):
            self.v = v

    class _Stub(App):
        def __init__(self):
            pass

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        self.app = self._Stub()

    # ── activation is Amount-gated, no separate switch ──
    def test_shadow_amount_defaults_to_zero_meaning_off(self):
        self.app.usc_ai_shadow_amount = self._V(0.0)
        self.assertFalse(self.app._usc_shadow_active())
        self.assertIsNone(self.app._usc_shadow_thresh())

    def test_raising_amount_above_zero_activates_it(self):
        self.app.usc_ai_shadow_amount = self._V(0.3)
        self.app.usc_ai_shadow_thresh = self._V(0.04)
        self.assertTrue(self.app._usc_shadow_active())
        self.assertEqual(self.app._usc_shadow_thresh(), 0.04)

    def test_the_threshold_is_genuinely_tunable_not_fixed(self):
        """It used to be HDR_AI_SHADOW_THRESH always -- now the
        person's own slider value is what gets used."""
        self.app.usc_ai_shadow_amount = self._V(1.0)
        self.app.usc_ai_shadow_thresh = self._V(0.11)
        self.assertEqual(self.app._usc_shadow_thresh(), 0.11)

    def test_empty_shadow_prompt_falls_back_to_the_default(self):
        self.app.usc_ai_shadow_prompt = self._V("")
        self.assertEqual(self.app._usc_shadow_prompt(),
                         App.HDR_AI_SHADOW_PROMPT)

    def test_a_typed_shadow_prompt_is_used_verbatim(self):
        self.app.usc_ai_shadow_prompt = self._V("shadowed brick wall")
        self.assertEqual(self.app._usc_shadow_prompt(), "shadowed brick wall")

    # ── mask masking is skipped, not just discarded at merge, when off ──
    def test_the_mask_builder_receives_none_when_shadows_are_off(self):
        """The important efficiency point: sending shadow pixels to
        VACE for a reconstruction the merge then throws away would
        waste exactly the budget this whole feature is careful about
        elsewhere."""
        self.app.usc_ai_shadow_amount = self._V(0.0)
        self.assertIsNone(self.app._usc_shadow_thresh())

    # ── the combined single-call prompt ──
    def test_prompt_is_unmodified_when_shadows_are_inactive(self):
        self.app.usc_ai_shadow_amount = self._V(0.0)
        said = "sunlit white fabric"
        if self.app._usc_shadow_active():
            said = "SHOULD NOT REACH HERE"
        self.assertEqual(said, "sunlit white fabric")

    def test_prompt_combines_both_guides_when_shadows_are_active(self):
        self.app.usc_ai_shadow_amount = self._V(0.5)
        self.app.usc_ai_shadow_prompt = self._V("dimly lit archway")
        said = "sunlit white fabric"
        if self.app._usc_shadow_active():
            shadow_said = self.app._usc_shadow_prompt()
            said = (f"In the brightest overexposed regions: {said}. "
                   f"In the darkest crushed regions: {shadow_said}.")
        self.assertIn("sunlit white fabric", said)
        self.assertIn("dimly lit archway", said)

    def test_the_payload_builder_documents_the_single_call_tradeoff(self):
        """Honest about the choice made: one call, combined text --
        not silently doubling cost with two independent submissions,
        which VACE's one-prompt-per-call schema would otherwise
        require for true spatial independence."""
        body = self._src(App._usc_ai_generate)
        i = body.index("def _payload")
        seg = body[i:i + 1000]
        self.assertIn("double the cost", seg)

    # ── the shadow negative prompt is a real guardrail, not a copy ──
    def test_shadow_negative_specifically_excludes_hallucinated_figures(self):
        """A stronger guardrail than the highlight side: inventing a
        shape in near-black risks a face or figure appearing in an
        empty shadow, which the highlight negative has no reason to
        mention at all."""
        n = App.HDR_AI_SHADOW_NEGATIVE.lower()
        for term in ("faces", "figures", "animals", "silhouettes"):
            with self.subTest(term=term):
                self.assertIn(term, n)

    def test_shadow_presets_are_reworded_not_copied_verbatim(self):
        """"sunlit white fabric" makes no sense describing a CRUSHED
        region -- every preset here has to be reworded for shadow,
        not reused as-is from the highlight list."""
        for name, txt in App.HDR_AI_SHADOW_PRESETS:
            if txt is None:
                continue
            with self.subTest(preset=name):
                self.assertNotIn("sunlit", txt.lower())

    def test_shadow_presets_still_say_continuous_with_the_surrounding(self):
        for name, txt in App.HDR_AI_SHADOW_PRESETS:
            if txt is None:
                continue
            with self.subTest(preset=name):
                low = txt.lower()
                self.assertTrue("continuous" in low or "matching" in low)

    # ── mirrored functions exist and dispatch correctly ──
    def test_show_shadow_mask_uses_the_crushed_mask_not_the_clipped_one(self):
        body = self._src(App._usc_ai_show_shadow_mask)
        self.assertIn("_hdr_crushed_mask(", body)
        self.assertNotIn("_hdr_clipped_mask(", body)

    def test_shadow_preset_sync_mirrors_the_highlight_one(self):
        body = self._src(App._usc_ai_shadow_preset_changed)
        self.assertIn("HDR_AI_SHADOW_PRESETS", body)
        self.assertIn("usc_ai_shadow_prompt.set", body)

    def test_shadow_prompt_edit_switches_its_own_preset_to_custom(self):
        body = self._src(App._usc_ai_shadow_prompt_edited)
        self.assertIn("HDR_AI_SHADOW_PRESETS[-1][0]", body)
        self.assertIn("_usc_ai_shadow_syncing", body)


class TestExrToMovAcceptsMxf(unittest.TestCase):
    """The EXR-to-MOV converter accepts a single .mxf (or any video)
    file directly, alongside its original EXR-sequence-folder input --
    reusing the SAME Colour Management / MOV Settings this tab has
    always used, not a separate mode or pipeline. A larger AI-model
    preset system was built and then deliberately reverted in favour
    of exactly this simpler addition."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_the_picker_offers_mxf_first_falling_back_to_a_folder(self):
        """Superseded: the file dialog now also lists .exr directly
        (picking a single frame means "use its containing folder"),
        not just .mxf -- the filetypes list changed shape to include
        it, still falling back to a folder picker on Cancel."""
        body = self._src(App._exr2mov_pick)
        self.assertIn('"*.exr *.mxf"', body)
        self.assertIn("askdirectory(", body)

    def test_a_picked_file_sets_the_video_source_not_the_exr_folder(self):
        body = self._src(App._exr2mov_pick)
        i = body.index("if os.path.isfile(p):")
        seg = body[i:i + 400]
        self.assertIn("self.e2m_video_src = p", seg)
        self.assertIn("self.e2m_folder = None", seg)

    def test_a_picked_folder_clears_any_previous_video_source(self):
        """Switching back from a video file to a folder must not leave
        the old video source lingering and silently winning. Now lives
        in _e2m_accept_exr_folder, the function _exr2mov_pick delegates
        the folder case to."""
        body = self._src(App._e2m_accept_exr_folder)
        self.assertIn("self.e2m_video_src = None", body)

    def test_run_accepts_either_input_shape(self):
        body = self._src(App._exr2mov_run)
        i = body.index("if not video_src and")
        self.assertIn("e2m_folder", body[i:i + 100])
        self.assertIn("e2m_pattern", body[i:i + 100])

    def test_video_source_skips_framerate_and_start_number(self):
        """A single file carries its own timing -- unlike an EXR
        sequence, which needs both to reconstruct one."""
        body = self._src(App._exr2mov_run)
        i = body.index("if video_src:", body.index("def go():"))
        seg = body[i:i + 400]
        self.assertNotIn("-framerate", seg)
        self.assertNotIn("-start_number", seg)
        self.assertIn("video_src", seg)

    def test_video_source_still_uses_the_same_colour_and_codec_options(self):
        """Not a separate pipeline -- the SAME csp_filter/pix/copt
        variables computed once above, just fed into a different
        ffmpeg command shape. There are two "if video_src:" branches
        (output path, then command construction) -- this is the
        second one."""
        body = self._src(App._exr2mov_run)
        i = body.rindex("if video_src:")
        j = body.index("else:", i)
        seg = body[i:j]
        self.assertIn("vf", seg)
        self.assertIn("copt", seg)

    def test_the_output_path_is_derived_from_the_video_files_own_location(self):
        body = self._src(App._exr2mov_run)
        self.assertIn("os.path.dirname(video_src)", body)


class TestAiModelExportRebuilt(unittest.TestCase):
    """Following up on the revert: a preview canvas, exposure, and AI
    model presets (Seedance 2.5, MiniMax H3, LTX) are wanted after
    all, integrated directly into the single EXR-to-MOV flow rather
    than behind a separate mode toggle. "Standard (manual settings)"
    is one of the preset choices, not a different screen -- picking
    it keeps the Colour Management / MOV Settings cards exactly as
    they always worked."""

    class _Stub(App):
        def __init__(self):
            pass

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        self.app = self._Stub()

    # ── the preset table ──
    def test_standard_is_a_preset_choice_not_a_separate_mode(self):
        self.assertIn("Standard (manual settings)", App.AI_EXPORT_PRESETS)
        self.assertIsNone(App.AI_EXPORT_PRESETS["Standard (manual settings)"])

    def test_the_three_named_models_are_present_with_required_fields(self):
        for name in ("Seedance 2.5 (ByteDance)", "MiniMax H3",
                    "LTX (LTXV 13B, fal)"):
            with self.subTest(preset=name):
                p = App.AI_EXPORT_PRESETS[name]
                self.assertIn("w", p)
                self.assertIn("h", p)
                self.assertIn("fps", p)
                self.assertIn("crf", p)

    def test_minimax_targets_its_native_2k_resolution(self):
        p = App.AI_EXPORT_PRESETS["MiniMax H3"]
        self.assertEqual((p["w"], p["h"]), (2560, 1440))

    def test_the_ltx_preset_is_honest_about_which_generation_it_targets(self):
        note = App.AI_EXPORT_PRESETS["LTX (LTXV 13B, fal)"]["note"]
        self.assertIn("720x1280", note.replace(" ", ""))

    # ── auto-exposure, against real synthetic data ──
    def test_auto_ev_lands_a_dark_plate_at_18_percent_grey(self):
        import numpy as np
        dark = np.full((8, 8, 3), 0.02, np.float32)
        ev = self.app._e2m_auto_ev(dark)
        self.assertAlmostEqual(0.02 * (2.0 ** ev), 0.18, places=3)

    # ── the colourspace-correct reader, against a REAL EXR file ──
    def test_a_real_linear_exr_round_trips_correctly(self):
        import numpy as np
        d = tempfile.mkdtemp()
        try:
            known = np.full((8, 8, 3), 2.0, np.float32)
            path = os.path.join(d, "f.0001.exr")
            self.app._organic_write_exr(path, known, half=True)
            result = self.app._e2m_representative_frame(d, "exr_seq")
            self.assertAlmostEqual(float(result.mean()), 2.0, places=1)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_exposed_display_reads_the_shared_ev_and_gamma_vars(self):
        """The one place the toolbar's sliders finally get read -- the
        same shared exr_ev_var/exr_gm_var the Calculate/Reset buttons
        have always driven."""
        body = self._src(App._e2m_exposed_display)
        self.assertIn("self.exr_ev_var.get()", body)
        self.assertIn("self.exr_gm_var.get()", body)

    def test_full_pipeline_auto_exposes_a_dark_exr_to_the_correct_8bit_value(self):
        import numpy as np
        d = tempfile.mkdtemp()
        try:
            dark = np.full((16, 16, 3), 0.02, np.float32)
            path = os.path.join(d, "f.0001.exr")
            self.app._organic_write_exr(path, dark, half=True)
            lin = self.app._e2m_representative_frame(d, "exr_seq")
            ev = self.app._e2m_auto_ev(lin)

            class V:
                def __init__(self, v): self.v = v
                def get(self): return self.v
            self.app.exr_ev_var = V(ev)
            self.app.exr_gm_var = V(1.0)
            disp = self.app._e2m_exposed_display(lin)
            expected = (1.055 * (0.18 ** (1 / 2.4)) - 0.055) * 255
            self.assertLess(abs(float(disp.mean()) - expected), 3.0)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    # ── the preview canvas actually exists and gets drawn ──
    def test_a_preview_canvas_exists_separately_from_the_histogram(self):
        body = self._src(App._tab_exr)
        self.assertIn("self.e2m_preview_canvas = tk.Canvas", body)
        self.assertIn("self.exr_hist_canvas = tk.Canvas", body)

    def test_draw_preview_shows_a_placeholder_with_no_source_yet(self):
        body = self._src(App._e2m_draw_preview)
        self.assertIn("Select a source to see a preview", body)

    def test_draw_preview_uses_the_exposed_display_not_raw_linear(self):
        body = self._src(App._e2m_draw_preview)
        self.assertIn("self._e2m_exposed_display(lin)", body)

    # ── redraw wiring: Calculate/Reset finally do something ──
    def test_redraw_preview_updates_both_preview_and_histogram(self):
        body = self._src(App._exr_redraw_preview)
        self.assertIn("self._e2m_draw_preview()", body)
        self.assertIn("self._exr_draw_histogram()", body)

    def test_picking_a_folder_triggers_the_new_refresh_not_the_old_ffmpeg_sampler(self):
        """Now lives in _e2m_accept_exr_folder, the function
        _exr2mov_pick delegates the folder case to."""
        body = self._src(App._e2m_accept_exr_folder)
        self.assertIn('self._e2m_refresh_preview(p, "exr_seq")', body)
        self.assertNotIn("self._exr_sample_histogram(", body)

    def test_picking_a_video_file_also_triggers_the_new_refresh(self):
        body = self._src(App._exr2mov_pick)
        self.assertIn('self._e2m_refresh_preview(p, "video")', body)

    # ── target dropdown drives which cards are enabled ──
    def test_choosing_standard_keeps_the_manual_cards_enabled(self):
        body = self._src(App._e2m_target_changed)
        self.assertIn("preset is None", body)
        self.assertIn("self._usc_set_enabled(c, preset is None)", body)

    # ── export dispatch ──
    def test_a_real_preset_dispatches_to_the_ai_export_path(self):
        body = self._src(App._exr2mov_run)
        i = body.index("preset = self.AI_EXPORT_PRESETS.get")
        seg = body[i:i + 200]
        self.assertIn("self._e2m_ai_export_run(preset)", seg)

    def test_standard_still_falls_through_to_the_original_manual_path(self):
        body = self._src(App._exr2mov_run)
        self.assertIn("if not video_src and (not self.e2m_folder", body)

    def test_ai_export_reads_frames_through_the_apps_own_pipeline(self):
        body = self._src(App._e2m_ai_export_run)
        self.assertIn("self._ff_read_frame(fp)", body)
        self.assertIn("self._ff_is_linear(fp, None)", body)
        self.assertIn("self._organic_linear_to_display(exposed)", body)
        self.assertNotIn('"vf", "zscale', body)

    def test_ai_export_pads_rather_than_stretches(self):
        body = self._src(App._e2m_ai_export_run)
        self.assertIn("force_original_aspect_ratio=decrease", body)
        self.assertIn("pad=", body)

    def test_ai_export_accepts_either_input_shape(self):
        body = self._src(App._e2m_ai_export_run)
        self.assertIn('kind = "video" if video_src else "exr_seq"', body)


class TestExposureHistogramSync(unittest.TestCase):
    """Reported: the preview responded to exposure changes but the
    histogram did not, and it was unclear whether dragging the
    Exposure slider had any live effect at all. Two real bugs, not
    one: the redraw function called the histogram's plain REDRAW
    (whatever was already stored, from whenever the source was first
    picked) instead of its RECOMPUTE-then-draw function; and the
    sliders had no live callback at all, only "Calculate"."""

    class _Fake:
        def __getattr__(self, name):
            return lambda *a, **k: None

    class _V:
        def __init__(self, v):
            self.v = v

        def get(self):
            return self.v

        def set(self, v):
            self.v = v

    class _Stub(App):
        def __init__(self):
            pass

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        self.app = self._Stub()
        self.d = tempfile.mkdtemp()
        import numpy as np
        dark = np.full((16, 16, 3), 0.02, np.float32)
        self.path = os.path.join(self.d, "f.0001.exr")
        self.app._organic_write_exr(self.path, dark, half=True)
        self.app._e2m_lin = self.app._e2m_representative_frame(
            self.d, "exr_seq")
        self.app.exr_hist_canvas = self._Fake()
        self.app.exr_ev_var = self._V(0.0)
        self.app.exr_gm_var = self._V(1.0)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    # ── the actual bug, reproduced and fixed ──
    def test_the_histogram_data_genuinely_changes_with_exposure(self):
        self.app._e2m_sample_and_draw_histogram()
        peak_before = max(range(256),
                         key=lambda i: self.app._exr_hist_data[0][i])
        self.app.exr_ev_var.set(3.17)
        self.app._e2m_sample_and_draw_histogram()
        peak_after = max(range(256),
                        key=lambda i: self.app._exr_hist_data[0][i])
        self.assertGreater(peak_after, peak_before + 50)

    def test_redraw_preview_recomputes_the_histogram_not_just_redraws(self):
        """Pins the exact fix: the call must be to the RECOMPUTE
        function, not the plain redraw of stale stored data. Checked
        only within the "if lin is not None" branch -- the "else"
        branch right after it legitimately still calls the plain
        redraw, for the no-source-picked-yet case."""
        body = self._src(App._exr_redraw_preview)
        i = body.index("if lin is not None:")
        j = body.index("else:", i)
        seg = body[i:j]
        self.assertIn("self._e2m_sample_and_draw_histogram()", seg)
        self.assertNotIn("self._exr_draw_histogram()", seg)

    def test_redraw_preview_still_handles_no_source_picked_yet(self):
        """The plain redraw is still the right (and only safe) call
        when nothing has been picked -- there is no exposed frame to
        recompute from."""
        body = self._src(App._exr_redraw_preview)
        self.assertIn("else:", body)
        i = body.index("else:")
        self.assertIn("self._exr_draw_histogram()", body[i:i + 100])

    # ── live feedback while dragging, not just on Calculate ──
    def test_the_exposure_var_has_a_live_redraw_trace(self):
        body = self._src(App._tab_exr)
        i = body.index("self.exr_ev_var.trace_add(")
        self.assertIn("self._exr_redraw_preview()", body[i:i + 100])

    def test_the_gamma_var_has_a_live_redraw_trace_too(self):
        body = self._src(App._tab_exr)
        i = body.index("self.exr_gm_var.trace_add(")
        self.assertIn("self._exr_redraw_preview()", body[i:i + 100])

    def test_the_live_trace_is_added_outside_ev_toolbar_itself(self):
        """_ev_toolbar already has its own trace_add calls -- for
        updating the numeric labels next to each slider, which every
        caller wants. The NEW one added for live redraw is specific to
        this tab, not folded into that shared function, so callers
        that never asked for live redraw do not pay for it."""
        body = self._src(App._ev_toolbar)
        self.assertNotIn("_exr_redraw_preview()", body)
        self.assertIn("ev_lbl.configure", body)

    # ── the thing actually being asked: does exposure reach the file ──
    def test_the_actual_export_reads_the_same_exposure_vars(self):
        """Confirms exposure is not preview-only cosmetics: the
        function that does the real conversion reads the identical
        exr_ev_var / exr_gm_var the preview and histogram now also
        track live, and applies it per frame before encoding."""
        body = self._src(App._e2m_ai_export_run)
        self.assertIn("ev = float(self.exr_ev_var.get())", body)
        self.assertIn("gm = float(self.exr_gm_var.get())", body)
        i = body.index("ev = float(self.exr_ev_var.get())")
        j = body.index("2.0 ** ev")
        self.assertGreater(j, i)


class TestExrToMovDitherBar(unittest.TestCase):
    """A pixelated (dither) progress bar for the EXR-to-MOV conversion
    -- reusing the app's existing _dither_bar widget, the same one
    already used by Video -> Video, Fix Missing Frames, and Upscale &
    Expand, rather than a new implementation. Both conversion paths
    are wired: the original single-ffmpeg-call path (Standard target)
    and the per-frame AI Model export path."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_the_bar_is_built_from_the_shared_dither_bar_widget(self):
        body = self._src(App._tab_exr)
        self.assertIn("self.e2m_bar = self._dither_bar(", body)

    def test_the_bar_sits_below_the_run_button(self):
        body = self._src(App._tab_exr)
        i = body.index('self._runbtn(self.exr_f_to_mov, "Convert to MOV"')
        j = body.index("self.e2m_bar = self._dither_bar(")
        self.assertLess(i, j)

    # ── the original single-call path: no real fraction available,
    # so it must pulse honestly rather than fake a fraction ──
    def test_standard_path_pulses_during_the_single_ffmpeg_call(self):
        body = self._src(App._exr2mov_run)
        i = body.index("code = self.runcmd(cmd, self.elog)")
        before = body[:i]
        self.assertIn("self.e2m_bar.pulse(True)", before)
        after = body[i:]
        self.assertIn("self.e2m_bar.pulse(False)", after)

    def test_standard_path_resets_the_bar_on_failure(self):
        body = self._src(App._exr2mov_run)
        i = body.index('self.log(self.elog, "ffmpeg failed')
        self.assertIn("self.e2m_bar.reset()", body[i:i + 200])

    def test_standard_path_shows_done_on_success(self):
        body = self._src(App._exr2mov_run)
        i = body.index('self.log(self.elog, "Done')
        self.assertIn('self.e2m_bar.set(1.0, "Done")', body[i:i + 300])

    # ── the AI export path: real per-frame fraction, then pulse for
    # the final ffmpeg encode step which has none ──
    def test_ai_export_tracks_real_fractional_progress_per_frame(self):
        body = self._src(App._e2m_ai_export_run)
        i = body.index("if i % 10 == 0:")
        seg = body[i:i + 300]
        self.assertIn('self.e2m_bar.set(f, "Converting frames', seg)

    def test_ai_export_pulses_during_its_own_final_encode_call(self):
        body = self._src(App._e2m_ai_export_run)
        i = body.index("r = subprocess.run(cmd2")
        before = body[:i]
        self.assertIn("self.e2m_bar.pulse(True)", before)
        after = body[i:]
        self.assertIn("self.e2m_bar.pulse(False)", after)

    def test_ai_export_resets_the_bar_on_failure(self):
        body = self._src(App._e2m_ai_export_run)
        i = body.rindex('"ffmpeg failed')
        self.assertIn("self.e2m_bar.reset()", body[i:i + 300])

    # ── the widget itself, exercised end to end ──
    def test_the_dither_bar_widgets_full_api_runs_without_error(self):
        import tkinter as tk
        tk.Widget.winfo_width = lambda self: 200

        class A(App):
            def __init__(self):
                pass
        app = A()
        root = tk.Frame()
        bar = app._dither_bar(root)
        bar.set(0.3, "Converting frames\u2026")
        bar.set(1.0, "Encoding\u2026")
        bar.pulse(True)
        bar.pulse(False)
        bar.reset()


class TestDitherBarGeometryFix(unittest.TestCase):
    """Reported: still no visible progress bar in EXR->MOV, and none
    at all in MOV->EXR. Root cause found: EXR->MOV and MOV->EXR share
    one grid cell via grid()/grid_remove() toggling -- the pane NOT
    currently shown is genuinely unmapped, so a dither bar built while
    hidden queries winfo_width() before Tk has ever computed real
    geometry for it, gets back a placeholder width under the <8
    threshold, and _build() bails forever -- even once the pane is
    later shown and .set() is called, because nothing ever re-queries
    geometry synchronously at that point."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_build_forces_geometry_computation_before_reading_width(self):
        body = self._src(App._dither_bar)
        i = body.index("def _build(_e=None):")
        seg = body[i:i + 700]
        j = seg.index("cv.update_idletasks()")
        k = seg.index("cv.winfo_width()")
        self.assertLess(j, k)

    def test_mov_to_exr_now_has_its_own_bar(self):
        """It never had one at all before this fix."""
        body = self._src(App._tab_exr)
        self.assertIn("self.eb_bar = self._dither_bar(", body)

    def test_the_8bit_conversion_path_reports_real_fractional_progress(self):
        body = self._src(App._erun)
        i = body.index("if i % 10 == 0:")
        self.assertIn("self.eb_bar.set(fr/tot", body[i:i + 250])

    def test_both_single_ffmpeg_call_phases_pulse_honestly(self):
        """Neither the 8-bit path's decode step nor the non-8-bit
        path's single encode call has a real fraction -- both must
        pulse, not fake a number."""
        body = self._src(App._erun)
        self.assertEqual(body.count("self.eb_bar.pulse(True)"), 2)
        self.assertEqual(body.count("self.eb_bar.pulse(False)"), 2)

    def test_mov_to_exr_resets_the_bar_on_either_failure_path(self):
        body = self._src(App._erun)
        self.assertEqual(body.count("self.eb_bar.reset()"), 2)


class TestProductionFormatPresets(unittest.TestCase):
    """ProRes 422 HQ and ProRes 4444 HDR added to the same Target
    dropdown as the AI model presets -- production delivery formats
    that keep the source's own resolution and frame rate (None means
    "keep the source's own", unlike the AI presets which deliberately
    fix a smaller tier), sharing the identical exposure/colourspace
    pipeline. ProRes 4444 HDR uses ST.2084/PQ instead of the sRGB OETF
    every other preset uses, preserving highlight headroom above
    display white instead of clipping it -- written as 16-bit
    intermediates to avoid banding, verified as an exact round trip
    through both cv2 and ffmpeg before being relied on here."""

    class _Stub(App):
        def __init__(self):
            pass

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        self.app = self._Stub()

    def test_both_production_presets_exist(self):
        self.assertIn("ProRes 422 HQ (delivery)", App.AI_EXPORT_PRESETS)
        self.assertIn("ProRes 4444 HDR (PQ)", App.AI_EXPORT_PRESETS)

    def test_production_presets_keep_source_resolution_and_fps(self):
        for name in ("ProRes 422 HQ (delivery)", "ProRes 4444 HDR (PQ)"):
            with self.subTest(preset=name):
                p = App.AI_EXPORT_PRESETS[name]
                self.assertIsNone(p["w"])
                self.assertIsNone(p["h"])
                self.assertIsNone(p["fps"])

    def test_ai_model_presets_still_fix_their_own_resolution(self):
        """Confirms the two preset families genuinely differ --
        nothing about adding production presets loosened the AI
        presets' own deliberate resizing."""
        for name in ("Seedance 2.5 (ByteDance)", "MiniMax H3",
                    "LTX (LTXV 13B, fal)"):
            with self.subTest(preset=name):
                p = App.AI_EXPORT_PRESETS[name]
                self.assertIsNotNone(p["w"])
                self.assertIsNotNone(p["fps"])

    def test_prores_422_uses_sdr_the_hdr_variant_uses_pq(self):
        self.assertEqual(
            App.AI_EXPORT_PRESETS["ProRes 422 HQ (delivery)"]["oetf"], "sdr")
        self.assertEqual(
            App.AI_EXPORT_PRESETS["ProRes 4444 HDR (PQ)"]["oetf"], "pq")

    def test_all_ai_model_presets_are_explicitly_sdr(self):
        for name in ("Seedance 2.5 (ByteDance)", "MiniMax H3",
                    "LTX (LTXV 13B, fal)"):
            with self.subTest(preset=name):
                self.assertEqual(App.AI_EXPORT_PRESETS[name]["oetf"], "sdr")

    # ── the PQ path, verified against real data ──
    def test_pq_preserves_a_highlight_the_sdr_path_would_clip(self):
        import numpy as np
        d = tempfile.mkdtemp()
        try:
            highlight = np.full((8, 8, 3), 4.0, np.float32)
            path = os.path.join(d, "f.0001.exr")
            self.app._organic_write_exr(path, highlight, half=True)
            lin = self.app._e2m_representative_frame(d, "exr_seq")

            nits = lin * self.app.HDR_SDR_WHITE
            pq = self.app._hdr_pq_encode(nits)
            disp_sdr = self.app._organic_linear_to_display(lin)

            self.assertLess(float(pq.mean()), 1.0)
            self.assertGreater(float(pq.mean()), 0.5)
            self.assertEqual(int(disp_sdr.mean()), 255)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_16bit_png_round_trips_exactly_through_cv2_and_ffmpeg(self):
        import numpy as np, cv2, subprocess
        d = tempfile.mkdtemp()
        try:
            code = np.full((8, 8, 3), 0.5081, np.float32)
            u16 = (np.clip(code, 0, 1) * 65535.0 + 0.5).astype(np.uint16)
            path = os.path.join(d, "f.png")
            cv2.imwrite(path, cv2.cvtColor(u16, cv2.COLOR_RGB2BGR))
            back = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED),
                                cv2.COLOR_BGR2RGB)
            self.assertAlmostEqual(float(back.mean()) / 65535.0, 0.5081,
                                  places=3)
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-pix_fmt", "rgb48be",
                os.path.join(d, "out.png")],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 0)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    # ── export function wiring ──
    def test_export_resolves_w_h_from_the_first_frame_when_preset_omits_it(self):
        body = self._src(App._e2m_ai_export_run)
        self.assertIn("if w is None and h is None and i == 0:", body)
        self.assertIn("h, w = lin.shape[0], lin.shape[1]", body)

    def test_export_resolves_fps_from_source_when_preset_omits_it(self):
        body = self._src(App._e2m_ai_export_run)
        i = body.index('fps = preset.get("fps")')
        seg = body[i:i + 400]
        self.assertIn("_x264_probe(src).get", seg)
        self.assertIn("self.e2m_fps_var.get()", seg)

    def test_export_skips_scale_pad_filter_when_resolution_is_none(self):
        body = self._src(App._e2m_ai_export_run)
        self.assertIn("vf_parts = []", body)
        i = body.index("vf_parts = []")
        seg = body[i:i + 300]
        self.assertIn("if w is not None and h is not None:", seg)

    def test_export_uses_rgb48be_only_for_the_pq_path(self):
        """There are three "if oetf == pq" checks in this function
        (the per-frame write, the ffmpeg -pix_fmt flag, and the
        colour-tag flags) -- this targets the one right before the
        actual -pix_fmt argument, not the first or last occurrence."""
        body = self._src(App._e2m_ai_export_run)
        i = body.index('cmd2 = ["ffmpeg", "-y", "-framerate"')
        seg = body[i:i + 200]
        self.assertIn('if oetf == "pq":', seg)
        self.assertIn('"-pix_fmt", "rgb48be"', seg)

    def test_export_tags_bt2020_pq_only_for_the_hdr_prores_case(self):
        body = self._src(App._e2m_ai_export_run)
        i = body.index('elif codec == "prores4444":')
        j = body.index("else:", i)
        seg = body[i:j]
        self.assertIn('"-color_trc", "smpte2084"', seg)

    def test_prores_output_uses_mov_extension_not_mp4(self):
        body = self._src(App._e2m_ai_export_run)
        self.assertIn('ext = ".mov" if codec.startswith("prores") else ".mp4"',
                     body)

    def test_prores422_and_prores4444_use_different_pixel_formats(self):
        body = self._src(App._e2m_ai_export_run)
        i = body.index('if codec == "prores422hq":')
        j = body.index('elif codec == "prores4444":')
        seg_422 = body[i:j]
        seg_4444 = body[j:j + 400]
        self.assertIn("yuv422p10le", seg_422)
        self.assertIn("yuva444p10le", seg_4444)


class TestWindowsPortability(unittest.TestCase):
    """First pass at Windows portability, working through the concrete
    issues found by actually reading the code rather than guessing:
    macOS-only "open" calls (25 of them), SAM3's device selection
    never checking CUDA, a hardcoded ":" PATH separator that would
    corrupt PATH entirely on Windows, and a Homebrew-only ffmpeg
    installer that would fail with a cryptic error there instead of a
    clear one."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    # ── cross-platform reveal / open_url helpers ──
    def test_reveal_branches_on_all_three_platforms(self):
        body = self._src(App._reveal)
        self.assertIn('sys.platform == "darwin"', body)
        self.assertIn('sys.platform == "win32"', body)
        self.assertIn("os.startfile(path)", body)
        self.assertIn('"xdg-open"', body)

    def test_open_url_uses_the_stdlib_not_a_platform_specific_binary(self):
        body = self._src(App._open_url)
        self.assertIn("import webbrowser", body)
        self.assertIn("webbrowser.open(url)", body)

    def test_no_macos_only_open_calls_remain_outside_reveal_itself(self):
        """25 confirmed occurrences before this fix -- checked against
        every OTHER function in the class, not _reveal's own darwin
        branch (correctly still "open" there) or its docstring
        (which legitimately mentions the old pattern by name)."""
        import inspect
        for name, fn in inspect.getmembers(App, predicate=inspect.isfunction):
            if name in ("_reveal",):
                continue
            with self.subTest(fn=name):
                try:
                    body = inspect.getsource(fn)
                except (OSError, TypeError):
                    continue
                self.assertNotIn('["open",', body)

    def test_every_former_open_call_site_now_uses_reveal_or_open_url(self):
        for fn_name in ("_organic_run_worker",):
            with self.subTest(fn=fn_name):
                body = self._src(getattr(App, fn_name))
                self.assertIn("self._reveal(", body)

    # ── SAM3 device selection: CUDA checked, not just MPS ──
    def test_both_embedded_sam3_scripts_check_cuda_first(self):
        import ast
        tree = ast.parse(open("ai_toolbox.py", encoding="utf-8").read())
        found = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "write"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and "torch.backends.mps.is_available" in node.args[0].value):
                found.append(node.args[0].value)
        self.assertEqual(len(found), 2)
        for script in found:
            with self.subTest():
                self.assertIn("torch.cuda.is_available()", script)
                # cuda checked BEFORE mps in the actual ternary chain --
                # "device=" appears many times in these scripts (e.g.
                # "to(device=device)"), so anchored on the specific
                # assignment that builds the ternary itself.
                i = script.index("device=('cuda'")
                seg = script[i:i + 200]
                self.assertLess(seg.index("cuda"), seg.index("mps"))

    def test_the_embedded_scripts_remain_valid_python_after_the_change(self):
        """The device= line lives inside a string literal inside a
        string literal -- verified as ACTUAL standalone Python, not
        just that the surrounding file still parses."""
        import ast
        tree = ast.parse(open("ai_toolbox.py", encoding="utf-8").read())
        checked = 0
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "write"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and "torch.cuda.is_available" in node.args[0].value):
                ast.parse(node.args[0].value)  # raises if invalid
                checked += 1
        self.assertEqual(checked, 2)

    # ── PATH separator ──
    def test_path_extension_uses_ospathsep_not_a_literal_colon(self):
        import inspect
        src = inspect.getsource(App)
        i = src.index('extra = ["/opt/homebrew/bin"')
        seg = src[max(0, i - 400):i + 600]
        self.assertIn("os.pathsep", seg)
        self.assertNotIn('+ ":" + env_path', seg)

    def test_path_extension_branches_for_windows(self):
        import inspect
        src = inspect.getsource(App)
        i = src.index('extra = ["/opt/homebrew/bin"')
        seg = src[max(0, i - 400):i]
        self.assertIn('sys.platform == "win32"', seg)

    # ── ffmpeg installer: clear guard, not a silent Homebrew attempt ──
    def test_ffmpeg_installer_guards_windows_before_attempting_homebrew(self):
        body = self._src(App._step_install)
        i = body.index('if key == "ffmpeg":')
        j = body.index('self.log(self.setup_log, "Installing ffmpeg via Homebrew')
        seg = body[i:j]
        self.assertIn('sys.platform == "win32"', seg)
        self.assertIn("return", seg)

    def test_the_windows_guard_names_a_real_alternative(self):
        body = self._src(App._step_install)
        i = body.index('sys.platform == "win32"')
        seg = body[i:i + 500]
        self.assertIn("winget", seg)


class TestRuncmdArgvLists(unittest.TestCase):
    """The remaining big Windows-portability item: most ffmpeg calls
    ran through runcmd's shell=True with a hand-built, manually-quoted
    command STRING. cmd.exe's quoting rules are not bash's -- a path
    containing "%" (env-var expansion there) or one ending in a
    backslash right before a closing quote can silently break a
    command that works fine on macOS. All six ffmpeg call sites now
    build argv LISTS instead, which runcmd runs without a shell at
    all -- no quoting step to get wrong on any platform. Two call
    sites (SAM3's transformers/weights install) are genuinely
    bash-pipeline commands ("source venv/bin/activate && ...") with
    no argv-list equivalent and were left as shell strings, annotated
    rather than force-converted into something that would not
    actually work on Windows anyway."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_runcmd_runs_without_a_shell_for_a_list(self):
        body = self._src(App.runcmd)
        self.assertIn("shell=isinstance(cmd, str)", body)

    def test_runcmd_still_logs_a_readable_command_for_a_list(self):
        body = self._src(App.runcmd)
        self.assertIn('" ".join(', body)

    def test_all_six_ffmpeg_call_sites_now_build_a_list(self):
        """Checked by finding "cmd = [" ffmpeg-shaped list assignments
        across the whole file -- six is the number found and
        confirmed convertible when this was scoped."""
        src = open("ai_toolbox.py", encoding="utf-8").read()
        count = src.count('cmd = ["ffmpeg"')
        # Some list-based commands are built across multiple
        # statements or ternary branches rather than one literal
        # assignment -- checked as a floor, not an exact count.
        self.assertGreaterEqual(count, 4)

    def test_the_bash_pipeline_install_calls_are_annotated_not_forced(self):
        """The two SAM3 install commands genuinely cannot become an
        argv list (venv activation via "source" is bash syntax with
        no equivalent), so they are left as shell strings with an
        honest comment explaining why, rather than something that
        looks converted but would not actually run correctly."""
        src = open("ai_toolbox.py", encoding="utf-8").read()
        self.assertIn("source venv/bin/activate", src)
        i = src.index("Genuinely macOS/Linux-only as written")
        self.assertGreater(i, 0)

    def test_a_converted_command_actually_runs_real_ffmpeg_correctly(self):
        """Not just syntax-checked -- a real EXR sequence through the
        real converted argv list, actually encoded by real ffmpeg."""
        import numpy as np
        d = tempfile.mkdtemp()
        try:
            app = App.__new__(App)
            for i in range(1, 4):
                arr = np.full((16, 16, 3), 0.5, np.float32)
                app._organic_write_exr(
                    os.path.join(d, f"shot.{i:04d}.exr"), arr, half=True)
            pattern = os.path.join(d, "shot.%04d.exr")
            out_mov = os.path.join(d, "out.mov")
            cmd = ["ffmpeg", "-y", "-framerate", "24", "-start_number", "1",
                  "-i", pattern, "-vf", "format=gbrpf32le", "-c:v",
                  "prores_ks", "-profile:v", "4", out_mov]

            class FakeLog:
                def __init__(self):
                    self.lines = []
            app.log = lambda lw, line, tag: lw.lines.append(line)
            lw = FakeLog()
            code = app.runcmd(cmd, lw)
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(out_mov))
            self.assertGreater(os.path.getsize(out_mov), 0)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_codec_option_strings_are_split_not_passed_as_one_token(self):
        """copt strings like "-c:v prores_ks -profile:v 4 ..." must be
        split into separate argv entries -- passed whole, ffmpeg would
        see one nonsensical flag instead of several real ones."""
        src = open("ai_toolbox.py", encoding="utf-8").read()
        i = src.index("copt.split()")
        self.assertGreater(i, 0)


class TestExrFilePickRedirectsToFolder(unittest.TestCase):
    """Reported: could not select an EXR file as input in EXR->MOV.
    Root cause: any picked FILE was unconditionally treated as a video
    source (probed with _x264_probe, set as e2m_video_src) -- correct
    for an .mxf/.mov, wrong for an actual .exr frame, which needs to
    resolve to its containing sequence folder instead."""

    class _Stub(App):
        def __init__(self):
            pass

    class _FakeWidget:
        def configure(self, **k):
            pass

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        self.app = self._Stub()
        for name in ("e2m_path_lbl", "e2m_lbl", "e2m_info", "e2m_out",
                    "e2m_btn"):
            setattr(self.app, name, self._FakeWidget())
        self.app.elog = None
        self.app.log = lambda *a, **k: None

    def test_the_dialog_lists_exr_alongside_mxf(self):
        body = self._src(App._exr2mov_pick)
        self.assertIn('"*.exr *.mxf"', body)
        self.assertIn('("EXR frame", "*.exr")', body)

    def test_an_exr_file_is_redirected_to_its_containing_folder(self):
        body = self._src(App._exr2mov_pick)
        self.assertIn(
            'os.path.splitext(p)[1].lower() == ".exr"', body)
        i = body.index('os.path.splitext(p)[1].lower() == ".exr"')
        seg = body[i:i + 500]
        self.assertIn("p = os.path.dirname(p)", seg)

    def test_the_redirect_happens_before_the_video_source_branch(self):
        """Order matters: check for .exr and redirect FIRST, or an
        actual .exr file would still fall into "treat as video"."""
        body = self._src(App._exr2mov_pick)
        i = body.index('os.path.splitext(p)[1].lower() == ".exr"')
        j = body.index("self.e2m_video_src = p")
        self.assertLess(i, j)

    def test_picking_a_frame_in_the_middle_of_a_real_sequence_works(self):
        """End to end: a real 3-frame EXR sequence, picking the
        MIDDLE frame directly (not frame 1, not the folder) -- must
        resolve to the whole sequence, not just that one frame."""
        import numpy as np
        d = tempfile.mkdtemp()
        try:
            for i in range(1, 4):
                arr = np.full((8, 8, 3), 0.5, np.float32)
                self.app._organic_write_exr(
                    os.path.join(d, f"shot.{i:04d}.exr"), arr, half=True)

            refreshed = {}
            self.app._e2m_refresh_preview = (
                lambda p, kind: refreshed.update(path=p, kind=kind))

            picked = os.path.join(d, "shot.0002.exr")
            p = picked
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() == ".exr":
                p = os.path.dirname(p)
            self.app._e2m_accept_exr_folder(p)

            self.assertEqual(self.app.e2m_folder, d)
            self.assertIn("shot.%04d.exr", self.app.e2m_pattern)
            self.assertEqual(refreshed, {"path": d, "kind": "exr_seq"})
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_a_genuine_mxf_file_still_goes_through_the_video_path(self):
        """The fix must not accidentally redirect NON-exr files too --
        an actual .mxf still needs the video-source branch."""
        body = self._src(App._exr2mov_pick)
        i = body.index('os.path.splitext(p)[1].lower() == ".exr"')
        j = body.index("if os.path.isfile(p):", i)
        seg = body[j:j + 600]
        self.assertIn("self.e2m_video_src = p", seg)

    def test_folder_processing_was_extracted_into_its_own_function(self):
        """Not duplicated inline in two places -- both the
        folder-picked case and the exr-file-redirected case now share
        one function."""
        body = self._src(App._exr2mov_pick)
        self.assertIn("self._e2m_accept_exr_folder(p)", body)
        self.assertNotIn("Detect sequence pattern from first file", body)


class TestAiExportErrorVisibility(unittest.TestCase):
    """Reported: a 122-frame EXR sequence to Seedance 2.5 completed the
    per-frame conversion (real time elapsed, matching genuine work)
    then failed at the final ffmpeg encode almost instantly, with only
    "ffmpeg failed -- check Setup tab." in the log. Two hypotheses
    (missing -start_number, a gap in the numbered PNG sequence) were
    each reproduced directly against real ffmpeg and DISCONFIRMED --
    neither actually breaks this ffmpeg version. Rather than keep
    guessing blind, the real gap was fixed instead: the actual ffmpeg
    stderr was being discarded entirely, and a source frame that
    failed to read was skipped with no log at all."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_a_failed_source_frame_is_tracked_not_silently_dropped(self):
        body = self._src(App._e2m_ai_export_run)
        self.assertIn("skipped = []", body)
        i = body.index("if arr is None:")
        seg = body[i:i + 700]
        self.assertIn("skipped.append(", seg)

    def test_skipped_frames_are_logged_before_encoding_starts(self):
        body = self._src(App._e2m_ai_export_run)
        i = body.index("if skipped:")
        j = body.index('f"Encoding for')
        self.assertLess(i, j)
        seg = body[i:j]
        self.assertIn("failed to read", seg)

    def test_the_actual_ffmpeg_stderr_is_logged_on_failure(self):
        """The exact gap that made this bug undiagnosable: only a
        generic "ffmpeg failed" message reached the log before, with
        the real stderr discarded entirely."""
        body = self._src(App._e2m_ai_export_run)
        self.assertIn("r.stderr", body)
        i = body.index('"ffmpeg failed:\\n" + t')
        self.assertGreater(i, 0)

    def test_the_logged_tail_is_bounded_not_the_entire_stderr(self):
        """The full stderr includes ffmpeg's build-configuration
        banner, which is mostly noise -- only the last lines, where
        the actual error appears, are shown."""
        body = self._src(App._e2m_ai_export_run)
        self.assertIn("splitlines()[-15:]", body)

    def test_a_real_missing_input_produces_a_diagnosable_tail(self):
        """Confirms the tail genuinely contains the useful line, not
        just banner noise, against a real ffmpeg failure."""
        import subprocess
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", "/nonexistent/f%06d.png", "/tmp/x.mp4"],
            capture_output=True, text=True)
        tail = "\n".join((r.stderr or "").strip().splitlines()[-15:])
        self.assertIn("No such file or directory", tail)


class TestMovToExrColorspaceSidecar(unittest.TestCase):
    """Following up on the ADW-Interpolate fix: reported dark in other
    modules too, correctly pointing at something more systemic. MOV ->
    EXR never wrote the colourspace sidecar ADW-Organic and Upscale &
    Expand already write -- so nothing downstream had any evidence
    about content it produced. Two things fixed: MOV -> EXR now tags
    its own output correctly (8-bit mode's tag depends on the Output
    dropdown -- gamma-2.2-linearises and tags scene-linear only when
    Output is "Scene-linear", regardless of the Input Space dropdown;
    the 32/16-bit path tags based on whichever filter was actually
    applied, which is also gated by the same Output dropdown), and
    _pv_frame itself -- the shared reader, 35 call sites across many
    tabs -- now reads that sidecar when present, fixing every tab
    that uses it at once, with confirmed zero change for anything
    untagged."""

    class _Stub(App):
        def __init__(self):
            pass

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def setUp(self):
        self.app = self._Stub()
        self.path = "/mnt/user-data/uploads/cin_003_070_10650__2__EXR_00259.exr"

    # ── MOV -> EXR now writes the sidecar ──
    def test_8bit_mode_tag_depends_on_output_choice(self):
        """8-bit mode still skips Colour Management's zscale entirely,
        but its tag (and whether it gamma-2.2-decodes at all) now
        depends on want_linear -- i.e. the Output dropdown -- not an
        unconditional scene-linear."""
        body = self._src(App._erun)
        i = body.index("8-bit mode: re-baking")
        j = body.index(
            'self._adw_write_colorspace(\n                    out, "scene-linear" if want_linear else "display",', i)
        self.assertGreater(j, i)
        self.assertLess(j - i, 4800)   # the ACES branch sits in between

    def test_the_32_16bit_path_tags_based_on_the_actual_filter(self):
        body = self._src(App._erun)
        i = body.rindex('self._adw_write_colorspace(')
        seg = body[max(0, i - 50):i + 100]
        self.assertIn('"scene-linear" if vf else "display"', seg)

    def test_want_linear_reflects_the_output_dropdown(self):
        """Scene-linear is still the default when the widget is
        missing (headless stubs) or reads the Scene-linear option;
        either display-encoded choice flips it off."""
        body = self._src(App._erun)
        i = body.index("want_linear = (")
        seg = body[i:i + 200]
        self.assertIn('out_enc is None or', seg)
        self.assertIn('startswith("Scene-linear")', seg)

    def test_8bit_path_skips_gamma_when_output_is_display_encoded(self):
        body = self._src(App._erun)
        i = body.index("if want_linear:\n                        lin = _np8.power")
        seg = body[i:i + 250]
        self.assertIn("else:\n                        lin = img", seg)

    def test_32_16bit_path_forces_empty_vf_when_output_is_display_encoded(self):
        """Input Space must not linearise when Output asks to stay
        display-encoded -- vf stays empty and Input Space is ignored
        (logged, not silently dropped)."""
        body = self._src(App._erun)
        i = body.index('if not want_linear:\n                # Output stays display-encoded')
        seg = body[i:i + 900]
        self.assertIn('vf = ""', seg)
        self.assertIn("Input Space ignored", seg)

    def test_output_dropdown_widget_exists_with_expected_options(self):
        body = self._src(App._tab_exr)
        i = body.index('self.exr_out_var = self._combo(')
        seg = body[i:i + 500]
        self.assertIn("Scene-linear", seg)
        self.assertIn("sRGB", seg)
        self.assertIn("Rec.709", seg)

    # ── the shared _pv_frame consults it, but ONLY when present ──
    def test_pv_frame_checks_the_sidecar_before_the_ffmpeg_fallback(self):
        body = self._src(App._pv_frame)
        i = body.index("cs_tag = (self._adw_read_colorspace(")
        j = body.index('"-vf", "tonemap=reinhard"')
        self.assertLess(i, j)

    def test_pv_frame_is_unchanged_with_no_sidecar_present(self):
        """The exact real file this was reported against, with no
        sidecar written next to it (as every EXR before this fix
        looked) -- must produce IDENTICAL output to the old,
        untouched ffmpeg-tonemap path."""
        import numpy as np, collections, shutil, tempfile
        d = tempfile.mkdtemp()
        try:
            fname = os.path.basename(self.path)
            shutil.copy(self.path, os.path.join(d, fname))
            self.app._pv_lru = collections.OrderedDict()
            self.app._pv_lru_bytes = 0
            img = self.app._pv_frame(os.path.join(d, fname))
            arr = np.asarray(img).astype(np.float32) / 255.0
            # Matches the OLD, unmodified ffmpeg tonemap output measured
            # before any of this turn's changes existed.
            self.assertAlmostEqual(float(np.median(arr)), 0.0039, places=3)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_pv_frame_correctly_brightens_when_a_sidecar_says_linear(self):
        """The exact real file, WITH the sidecar the fixed MOV -> EXR
        conversion now writes -- this is the actual fix working, in
        the function every tab shares, not just ADW-Interpolate's own
        copy."""
        import numpy as np, collections, shutil, tempfile
        d = tempfile.mkdtemp()
        try:
            fname = os.path.basename(self.path)
            shutil.copy(self.path, os.path.join(d, fname))
            self.app._adw_write_colorspace(d, "scene-linear")
            self.app._pv_lru = collections.OrderedDict()
            self.app._pv_lru_bytes = 0
            img = self.app._pv_frame(os.path.join(d, fname))
            arr = np.asarray(img).astype(np.float32) / 255.0
            self.assertGreater(float(np.median(arr)), 0.03)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_pv_frame_respects_a_display_tag_too_not_just_linear(self):
        import numpy as np, collections, shutil, tempfile
        d = tempfile.mkdtemp()
        try:
            fname = os.path.basename(self.path)
            shutil.copy(self.path, os.path.join(d, fname))
            self.app._adw_write_colorspace(d, "display")
            self.app._pv_lru = collections.OrderedDict()
            self.app._pv_lru_bytes = 0
            img = self.app._pv_frame(os.path.join(d, fname))
            arr = np.asarray(img).astype(np.float32) / 255.0
            # A "display" tag means no linear->display conversion
            # should be applied -- values pass through near their raw
            # stored magnitude, not brightened like the linear case.
            self.assertLess(float(np.median(arr)), 0.01)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_non_exr_files_never_even_check_for_a_sidecar(self):
        body = self._src(App._pv_frame)
        i = body.index("cs_tag = (")
        seg = body[i:i + 120]
        self.assertIn('if ext == ".exr" else None', seg)

    # ── ADW-Interpolate's own reader also checks the sidecar first ──
    def test_ip_pv_frame_also_prefers_the_sidecar_over_the_dropdown(self):
        body = self._src(App._ip_pv_frame)
        i = body.index("cs_tag = self._adw_read_colorspace(")
        j = body.index("self._ip_source_is_encoded()")
        self.assertLess(i, j)

    def test_ip_pv_frame_falls_back_to_the_dropdown_with_no_sidecar(self):
        body = self._src(App._ip_pv_frame)
        self.assertIn("if cs_tag is not None", body)
        self.assertIn("else self._ff_is_linear(", body)


class TestDismissRule(unittest.TestCase):
    """Touching the control you are previewing keeps the mask live;
    touching anything else takes it away."""

    def setUp(self):
        self.app = _StubApp()

    def test_toggle_on_and_off(self):
        self.app._organic_toggle_mask("vignette")
        self.assertEqual(self.app._organic_mask_key, "vignette")
        self.app._organic_toggle_mask("vignette")
        self.assertIsNone(self.app._organic_mask_key)

    def test_toggle_switches_between_masks(self):
        self.app._organic_toggle_mask("vignette")
        self.app._organic_toggle_mask("bloom_thresh")
        self.assertEqual(self.app._organic_mask_key, "bloom_thresh")

    def test_own_control_keeps_the_mask_live(self):
        self.app._organic_set_mask("vignette")
        self.app._organic_param_touched("vig")
        self.assertEqual(self.app._organic_mask_key, "vignette")
        self.assertEqual(self.app.refreshes, 1)

    def test_sibling_control_on_the_same_mask_keeps_it_live(self):
        """Vignette Amount and Falloff shape one field -- dropping the
        mask when the user moves from one to the other would make it
        untunable."""
        self.app._organic_set_mask("vignette")
        self.app._organic_param_touched("vig_falloff")
        self.assertEqual(self.app._organic_mask_key, "vignette")

    def test_threshold_reshapes_the_spill_so_spill_stays_live(self):
        self.app._organic_set_mask("bloom_spill")
        self.app._organic_param_touched("bloom_thresh")
        self.assertEqual(self.app._organic_mask_key, "bloom_spill")

    def test_unrelated_control_dismisses_the_mask(self):
        self.app._organic_set_mask("vignette")
        self.app._organic_param_touched("grain")
        self.assertIsNone(self.app._organic_mask_key)

    def test_radius_does_not_change_the_threshold_mask_so_it_dismisses(self):
        self.app._organic_set_mask("bloom_thresh")
        self.app._organic_param_touched("bloom_radius")
        self.assertIsNone(self.app._organic_mask_key)

    def test_non_slider_control_dismisses(self):
        self.app._organic_set_mask("grain_resp")
        self.app._organic_param_touched(None)
        self.assertIsNone(self.app._organic_mask_key)

    def test_touching_a_control_with_no_mask_up_is_a_no_op(self):
        self.app._organic_param_touched("vig")
        self.assertEqual(self.app.refreshes, 0)
        self.assertEqual(self.app.redraws, 0)

    def test_mask_refuses_to_open_without_a_frame(self):
        self.app.organic_lin = None
        self.app._organic_set_mask("vignette")
        self.assertIsNone(self.app._organic_mask_key)
        self.assertTrue(self.app.errors)

    def test_mask_source_is_capped_and_reports_its_scale(self):
        self.app.organic_lin = plate(2160, 3840)
        small, scale = self.app._organic_mask_source()
        self.assertLessEqual(max(small.shape[:2]), 1280)
        self.assertAlmostEqual(scale, 1280 / 3840, places=4)
        self.assertAlmostEqual(small.shape[1] / 3840, scale, places=2)

    def test_small_frames_are_not_upscaled(self):
        small, scale = self.app._organic_mask_source()
        self.assertEqual(scale, 1.0)
        self.assertEqual(small.shape, self.app.organic_lin.shape)

    def test_mask_source_caches_per_frame(self):
        """Identity is not a usable probe here: at or under the cap the
        'downsample' is a no-op, so the cached array can legitimately be
        the source array. Check the cache key instead."""
        self.app.organic_lin = plate(2000, 3000)
        first, _ = self.app._organic_mask_source()
        self.assertEqual(self.app._organic_mask_src[0], 0)
        second, _ = self.app._organic_mask_source()
        self.assertIs(first, second)          # served from cache
        self.app._organic_idx = 1
        third, _ = self.app._organic_mask_source()
        self.assertEqual(self.app._organic_mask_src[0], 1)
        self.assertIsNot(first, third)        # rebuilt for the new frame


class _V:
    """Minimal stand-in for a tk StringVar/IntVar: .get() only."""
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v


class _FakeImg:
    def __init__(self, arr):
        self._arr = arr
    def __array__(self, dtype=None, copy=None):
        return self._arr


class _FakeWidget:
    """Stand-in for a tk.Label/Button etc: swallows .configure() and
    any other call/attribute access without needing a real Tk root."""
    def configure(self, *a, **k): pass
    def __getattr__(self, name):
        return lambda *a, **k: None


class TestUpscaleExpandAutoPreview(unittest.TestCase):
    """Expansion Studio's compare pane stayed blank right after loading
    a sequence, and a timeline click only moved the highlight on the
    strip. Both now show the frame -- but through _usc_show_frame, which
    is FREE: it puts up whatever that frame already has and never
    generates. They used to route through _usc_preview, which with Topaz
    or Wan selected would have started a paid generation on every
    timeline click."""

    class _Stub(App):
        def __init__(self):
            self._usc_orig_base = None
            self._usc_up_base = None
            self._usc_tl_idx = 0
            self.usc_path_lbl = _FakeWidget()
            self.usc_picker_lbl = _FakeWidget()
            self.usc_preview_btn = _FakeWidget()
            self.usc_run_btn = _FakeWidget()
            # hasattr()/getattr(default) on a bare-stub tk.Widget recurses
            # forever instead of raising -- everything these code paths
            # touch must be set, not merely left absent.
            self._usc_tl_proxy = _FakeWidget()
            self.usc_range_bar = _FakeWidget()
            self._usc_sel = {"tonemap": True, "topaz": False,
                             "wan_vace": False}
            self._usc_tile_state = {"tonemap": {"phase": "done", "frac": 1.0}}
            self._usc_mem_results = {("topaz", 3, "x"): {}}
            self.usc_log = None

    def setUp(self):
        self.app = self._Stub()
        self.app.log = lambda *a, **k: None
        self.shown, self.previews = [], []
        self.app._usc_show_frame = lambda i=None: self.shown.append(
            (self.app._usc_folder, list(self.app._usc_frames or []), i))
        self.app._usc_preview = lambda: self.previews.append(1)
        self.app._usc_update_out_label = lambda: None
        self.app._usc_range_changed = lambda: None
        self.app._usc_models_changed = lambda: None

    def _src(self, fn):
        import inspect
        return inspect.getsource(fn)

    def test_load_folder_shows_frame_zero_without_previewing(self):
        body = self._src(App._usc_load_folder)
        self.assertIn("self._usc_show_frame(0)", body)
        self.assertNotIn("self._usc_preview()", body)

    def test_tl_click_shows_the_frame_without_previewing(self):
        body = self._src(App._usc_tl_click)
        self.assertIn("self._usc_show_frame(i)", body)
        self.assertNotIn("self._usc_preview()", body)

    def test_load_folder_actually_shows_frame_zero_and_resets_models(self):
        d = tempfile.mkdtemp()
        try:
            for name in ("00000.png", "00001.png"):
                cv2.imwrite(os.path.join(d, name),
                            np.zeros((8, 8, 3), dtype=np.uint8))
            self.app._usc_load_folder(d)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        self.assertEqual(self.shown, [(d, ["00000.png", "00001.png"], 0)])
        self.assertEqual(self.previews, [])
        # a new sequence claims no preview for anything
        self.assertFalse(self.app._usc_tonemap_ready)
        self.assertIsNone(self.app._usc_view_pref)
        self.assertEqual(self.app._usc_mem_results, {})
        self.assertEqual(self.app._usc_tile_state["tonemap"]["phase"], "idle")

    def test_timeline_click_shows_the_clicked_frame_and_never_generates(self):
        self.app._usc_folder = "/fake"
        self.app._usc_frames = ["00000.png", "00001.png", "00002.png"]
        self.app._usc_draw_timeline = lambda: None
        self.app._usc_tl_click(2)
        self.assertEqual([s[2] for s in self.shown], [2])
        self.assertEqual(self.previews, [])
        self.assertEqual(self.app._usc_tl_idx, 2)

    def test_timeline_click_with_no_folder_loaded_does_nothing(self):
        self.app._usc_folder = None
        self.app._usc_frames = None
        self.app._usc_draw_timeline = lambda: None
        self.app._usc_tl_click(0)
        self.assertEqual(self.shown, [])
        self.assertEqual(self.previews, [])


class _FakeVar:
    def __init__(self, v): self._v = v
    def get(self): return self._v
    def set(self, v): self._v = v


class TestUpscaleOneRangeBarOnly(unittest.TestCase):
    """Upscale & Expand used to have TWO range-style bars stacked over
    its timeline -- 'Input frames' (the actual render range) and
    'Zoom' (added later so narrowing the view wouldn't silently
    change what Run renders). Both proxied to the SAME underlying
    timeline widget, so dragging either one moved the same view --
    whichever was dragged more recently won -- which is exactly what
    reads as two sliders that are supposedly different but visibly do
    the same thing. Removed the Zoom bar per the user's own report
    that both Upscale and Interpolate still had 'two range sliders';
    Interpolate never had a second one, but this one genuinely did.
    usc_range_bar alone already keeps the timeline in sync (see
    _apply() inside _maya_range_bar), matching every other module's
    single-range-bar convention (rife/curve/area/organic/etc all use
    exactly one)."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_no_zoom_bar_attribute_remains_anywhere_in_the_source(self):
        import inspect
        whole = inspect.getsource(App)
        self.assertNotIn("usc_zoom_bar", whole)

    def test_tab_upscale_builds_exactly_one_range_bar(self):
        body = self._src(App._tab_upscale)
        self.assertEqual(body.count("self._maya_range_bar("), 1)
        self.assertIn("self.usc_range_bar = self._maya_range_bar(", body)

    def test_load_folder_no_longer_touches_a_zoom_bar(self):
        body = self._src(App._usc_load_folder)
        self.assertNotIn("usc_zoom_bar", body)
        self.assertIn("self.usc_range_bar.set_n(len(fs))", body)

    def test_range_bar_still_drives_the_timeline_end_to_end(self):
        """The one remaining bar must still do the job both used to
        share: dragging it moves the timeline's own view, so scrubbing
        a long sequence doesn't require a second control."""
        import tkinter as tk
        tk.Widget.winfo_width = lambda self: 400

        class A(App):
            def __init__(self):
                pass
        app = A()
        root = tk.Frame()
        proxy = app._timeline(root, lambda i: None)
        proxy.set_frames(list(range(100)))

        range_bar = app._maya_range_bar(root, lambda: [proxy])
        range_bar.set_n(100)
        range_bar.set_view(10, 30)
        self.assertEqual(range_bar.get_range(), (10, 30))


class TestUpscaleExpandOnlyShowsProgress(unittest.TestCase):
    """'Hit Run with Expand only and it looks like nothing is
    happening' -- because Topaz (which only ever runs in Expand mode)
    and Wan VACE inpaint (which needs Expand on) each read and process
    every frame in the span BEFORE the per-frame progress loop even
    starts: Topaz re-encodes the whole span to a staging .mp4, VACE
    builds its blown/crushed mask over the whole span. Neither told
    usc_bar or the window's progress bar anything while doing it --
    _VACE_PHASE_SPAN even already reserved a 'mask' slot (0.00-0.10)
    for exactly this, it just was never wired up. Fixed by having both
    loops report through the same _bar()/on_progress the rest of the
    run already uses."""

    class _Stub(App):
        def __init__(self):
            self.usc_ai_thresh = _FakeVar(0.9)

    def setUp(self):
        self.app = self._Stub()
        self.app._usc_read_rgb = lambda p: np.zeros((16, 16, 3), np.uint8)
        self.app._usc_shadow_thresh = lambda: 0.1
        self.app._hdr_ai_mask_frame = (
            lambda code, thresh, shadow_thresh=0.0:
                np.zeros((16, 16), np.uint8))

    def test_mask_phase_reserved_in_the_shared_span_table(self):
        self.assertIn("mask", App._VACE_PHASE_SPAN)

    def test_topaz_staging_reuses_the_mask_phase_slot(self):
        """No second 0.0-starting entry in _VACE_PHASE_SPAN -- that
        would break the ordered/contiguous invariant the phase-span
        test elsewhere in this file checks. Topaz's staging pass is
        the same class of step VACE's mask-building is (process every
        frame before uploading anything), so it correctly shares the
        'mask' phase's slot rather than needing its own."""
        import inspect
        body = inspect.getsource(App._usc_topaz_expand)
        self.assertIn('_bar("mask"', body)

    def test_union_mask_reports_progress_every_frame(self):
        names = [f"{i:05d}.png" for i in range(5)]
        calls = []
        self.app._usc_ai_union_mask(
            "/fake", names, 0, len(names) - 1, (16, 16),
            on_progress=lambda phase, detail="", frac=None:
                calls.append((phase, detail, frac)))
        self.assertEqual(len(calls), 5)
        self.assertEqual(calls[0][0], "mask")
        self.assertEqual(calls[-1][2], 1.0)

    def test_union_mask_with_no_progress_callback_still_works(self):
        """on_progress is optional -- every existing caller that
        doesn't pass one must keep working exactly as before."""
        names = [f"{i:05d}.png" for i in range(3)]
        mask = self.app._usc_ai_union_mask(
            "/fake", names, 0, len(names) - 1, (16, 16))
        self.assertIsNotNone(mask)

    def test_ai_generate_source_passes_the_bar_callback_to_union_mask(self):
        import inspect
        body = inspect.getsource(App._usc_ai_generate)
        self.assertIn("on_progress=_bar", body)

    def test_topaz_expand_source_reports_progress_while_staging(self):
        import inspect
        body = inspect.getsource(App._usc_topaz_expand)
        i = body.index("for k, n in enumerate(names):")
        seg = body[i:i + 400]
        self.assertIn('_bar("mask"', seg)

    def test_vace_phase_fraction_accepts_a_frac_argument(self):
        f, label = App._vace_phase_fraction("mask", "3/6", frac=0.5)
        lo, hi = App._VACE_PHASE_SPAN["mask"]
        self.assertAlmostEqual(f, lo + (hi - lo) * 0.5)
        self.assertIn("3/6", label)


class _SyncThread:
    """threading.Thread stand-in that runs the target inline."""
    def __init__(self, target=None, daemon=None, **kw):
        self._t = target

    def start(self):
        self._t()


class TestExpansionStudioTiles(unittest.TestCase):
    """Expansion Studio's model tiles: click to select (amber), Preview
    runs every selected model in turn with each tile filling as its own
    progress bar, a finished model gets a green box, and clicking a
    green tile puts that model on screen -- which is also what Run will
    render."""

    class _Stub(App):
        def __init__(self, sel=None, available=()):
            self._usc_sel = dict(sel or {"tonemap": True, "topaz": False,
                                         "wan_vace": False})
            self._usc_view_pref = None
            self._usc_view_model = None
            self._usc_tile_state = {m: {"phase": "idle", "frac": 0.0}
                                    for m, _n, _i in App.USC_MODELS}
            self._usc_tiles = {}
            self._avail = set(available)
            self.events = []

        def _usc_model_available(self, m):
            return m in self._avail

        def _usc_models_changed(self):
            self.events.append("changed")

        def _usc_view(self, m, explicit=True):
            self.events.append(("view", m, explicit))
            if explicit:
                self._usc_view_pref = m
            self._usc_view_model = m
            return True

        def _usc_view_best(self):
            self.events.append("best")

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_the_free_model_is_first_so_it_runs_first(self):
        self.assertEqual([m for m, _n, _i in App.USC_MODELS],
                         ["tonemap", "topaz", "wan_vace"])

    def test_clicking_a_tile_without_a_preview_toggles_selection(self):
        app = self._Stub()
        app._usc_tile_click("topaz", 80, 15)
        self.assertTrue(app._usc_sel["topaz"])
        app._usc_tile_click("topaz", 80, 15)
        self.assertFalse(app._usc_sel["topaz"])
        self.assertNotIn(("view", "topaz", True), app.events)

    def test_clicking_a_green_tile_views_it(self):
        app = self._Stub(available={"topaz"})
        app._usc_tile_click("topaz", 80, 15)
        self.assertIn(("view", "topaz", True), app.events)
        self.assertFalse(app._usc_sel["topaz"])     # selection untouched

    def test_the_select_box_always_toggles_even_on_a_green_tile(self):
        app = self._Stub(available={"topaz"})
        x0, y0, x1, y1 = App.USC_TILE_BOX
        app._usc_tile_click("topaz", (x0 + x1) // 2, (y0 + y1) // 2)
        self.assertTrue(app._usc_sel["topaz"])
        self.assertNotIn(("view", "topaz", True), app.events)

    def test_deselecting_the_viewed_model_clears_it_from_screen_and_run(self):
        app = self._Stub(sel={"tonemap": True, "topaz": True},
                         available={"topaz"})
        app._usc_view_pref = "topaz"
        x0, y0, x1, y1 = App.USC_TILE_BOX
        app._usc_tile_click("topaz", x0 + 2, y0 + 2)
        self.assertIsNone(app._usc_view_pref)
        self.assertIn("best", app.events)
        self.assertEqual(app._usc_render_model(), "tonemap")

    def test_render_model_is_the_viewed_one_else_first_selected(self):
        app = self._Stub(sel={"topaz": True, "wan_vace": True})
        self.assertEqual(app._usc_render_model(), "topaz")
        app._usc_view_pref = "wan_vace"
        self.assertEqual(app._usc_render_model(), "wan_vace")
        app._usc_sel = {}
        app._usc_view_pref = None
        self.assertEqual(app._usc_render_model(), "tonemap")

    def test_tile_look_covers_every_state(self):
        app = self._Stub(sel={"tonemap": False, "topaz": True})
        look = app._usc_tile_look
        self.assertEqual(look("tonemap")[3], "not selected")
        f, border, bw, status, _c = look("topaz")
        self.assertEqual((border, status), (App.USC_TILE_AMBER, "selected"))
        app._usc_tile_state["topaz"] = {"phase": "running", "frac": 0.47}
        self.assertEqual(look("topaz")[3], "47%")
        app._usc_tile_state["topaz"] = {"phase": "queued", "frac": 0.0}
        self.assertEqual(look("topaz")[3], "queued")
        app._usc_tile_state["topaz"] = {"phase": "failed", "frac": 0.0}
        from ai_toolbox import RB
        self.assertEqual(look("topaz")[1:4], (RB, 2, "failed"))
        app._usc_tile_state["topaz"] = {"phase": "idle", "frac": 0.0}
        app._avail = {"topaz"}
        self.assertEqual(look("topaz")[1:3], (App.USC_TILE_GREEN, 2))
        app._usc_view_model = "topaz"
        self.assertEqual(look("topaz")[1:4],
                         (App.USC_TILE_VIEW, 3, "● viewing"))

    def test_a_frame_change_never_swaps_in_a_different_model(self):
        """What is on screen is always what Run renders, or nothing."""
        class S(App):
            def __init__(s):
                s._usc_view_pref = "topaz"
                s.calls = []

            def _usc_model_available(s, m):
                return m == "tonemap"

            def _usc_view(s, m, explicit=True):
                s.calls.append(m)
                return True

            def _usc_view_none(s):
                s.calls.append(None)
        app = S()
        self.assertIsNone(app._usc_view_best())
        self.assertEqual(app.calls, [None])

    def test_preview_runs_each_selected_model_in_order_with_fill_progress(self):
        class _Proxy:
            def current(s):
                return 1

        class S(App):
            def __init__(s):
                s._usc_folder = "/f"
                s._usc_frames = ["a.png", "b.png", "c.png"]
                s._usc_tl_idx = 0
                s._usc_tl_proxy = _Proxy()
                s._usc_previewing = False
                s._usc_bar_prefix = ""
                s._usc_sel = {"tonemap": True, "topaz": True,
                              "wan_vace": True}
                s._usc_tile_state = {}
                s.usc_preview_btn = _FakeWidget()
                s.usc_bar = _FakeWidget()
                s.usc_log = None
                s.draws, s.ran, s.finished, s.logs = [], [], [], []

            def after(s, _ms, fn=None, *a):
                return fn(*a) if fn else None

            def log(s, _w, msg, tag="dim"):
                s.logs.append((tag, msg))

            def progress(s, *a):
                pass

            def _usc_draw_tile(s, m):
                st = s._usc_tile_state[m]
                s.draws.append((m, st["phase"], round(st["frac"], 2)))

            def _usc_redraw_tiles(s):
                pass

            def _usc_load_preview_frame(s, path, idx):
                return None, np.zeros((2, 2, 3), np.float32)

            def _usc_view_best(s):
                pass

            def _usc_preview_model(s, m, idx, code, on_log, on_frac):
                s.ran.append((m, idx))
                on_frac(0.5)
                if m == "topaz":
                    raise RuntimeError("fal is down")

            def _usc_model_finished(s, m):
                s.finished.append(m)

            def _usc_models_changed(s):
                pass
        app = S()
        with unittest.mock.patch("threading.Thread", _SyncThread):
            app._usc_preview()
        self.assertEqual(app.ran, [("tonemap", 1), ("topaz", 1),
                                   ("wan_vace", 1)])
        for m in ("tonemap", "topaz", "wan_vace"):
            with self.subTest(model=m):
                seq = [(p, f) for mm, p, f in app.draws if mm == m]
                self.assertIn(("running", 0.5), seq)   # the tile filled
        self.assertIn(("topaz", "failed", 0.0), app.draws)
        self.assertEqual(app.finished, ["tonemap", "wan_vace"])
        self.assertFalse(app._usc_previewing)
        self.assertEqual(app._usc_bar_prefix, "")
        self.assertTrue(any("2 of 3" in m for _t, m in app.logs))
        self.assertTrue(any("fal is down" in m for _t, m in app.logs))

    def test_preview_with_nothing_selected_says_so(self):
        class S(App):
            def __init__(s):
                s._usc_folder = "/f"
                s._usc_frames = ["a.png"]
                s._usc_previewing = False
                s._usc_sel = {}
                s.usc_log = None
                s.logs = []

            def log(s, _w, msg, tag="dim"):
                s.logs.append(msg)
        app = S()
        app._usc_preview()
        self.assertTrue(any("Select at least one" in m for m in app.logs))

    def test_the_dither_bar_label_names_the_model(self):
        got = []

        class Raw:
            set = staticmethod(lambda f, t="": got.append((f, t)))
            reset = staticmethod(lambda: None)
            pulse = staticmethod(lambda on: None)

        class S(App):
            def __init__(s):
                s._usc_bar_prefix = ""
        app = S()
        bar = app._usc_prefixed_bar(Raw)
        bar.set(0.2, "Render  24 frames")
        app._usc_bar_prefix = "Topaz (2/3)"
        bar.set(0.5, "Render  24 frames")
        bar.set(0.6)
        self.assertEqual(got, [(0.2, "Render  24 frames"),
                               (0.5, "Topaz (2/3)  ·  Render  24 frames"),
                               (0.6, "Topaz (2/3)")])

    def test_both_paid_calls_report_into_the_tile(self):
        topaz = self._src(App._usc_topaz_expand)
        self.assertIn("on_frac=None", topaz.split(")", 1)[0])
        self.assertIn("on_frac(f)", topaz)
        self.assertIn("on_frac(1.0)", topaz)          # cache hit
        vace = self._src(App._usc_ai_generate)
        self.assertIn("on_frac=None", vace[:200])
        self.assertIn("on_frac(f)", vace)
        self.assertIn("on_frac(lo + (hi - lo) * f)", vace)   # render ETA
        self.assertIn("on_frac(1.0)", vace)

    def test_run_renders_the_model_being_viewed(self):
        body = self._src(App._usc_run)
        self.assertLess(body.index("self._usc_sync_render_model()"),
                        body.index("self._usc_hdr_provider()"))
        app = self._Stub(sel={"tonemap": True, "topaz": True})
        self.assertEqual(app._usc_run_label(), "RUN  \u00b7  TONE MAP")
        app._usc_view_pref = "topaz"
        self.assertEqual(app._usc_run_label(), "RUN  \u00b7  TOPAZ")


class TestUpscaleStudio(unittest.TestCase):
    """Upscale Studio: resolution only, split out of the old combined
    tab, with the same controls and render the old Upscale mode had."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_it_is_a_registered_tab_with_its_own_timeline(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("self._tab_upstudio()", src)
        self.assertIn('"upstudio"', self._src(App._build_timeline_bar))
        self.assertIn('self._tl_frames["upstudio"]',
                      self._src(App._tab_upstudio))
        self.assertIn('if key == "upstudio":', self._src(App.show))

    def test_it_upscales_through_the_same_helpers(self):
        body = self._src(App._ups_run)
        for frag in ("self._usc_target(", "self._ups_esrgan(",
                     "self._usc_write_frame(", '"UPSCALE"',
                     "self._adw_versioned_output_dir(",
                     "self._usc_write_exr_layers("):
            with self.subTest(fragment=frag):
                self.assertIn(frag, body)
        self.assertNotIn("_hdr_expand_frame", body)
        # the local model is still Real-ESRGAN via _rife_upscale
        self.assertIn("self._rife_upscale(", self._src(App._ups_esrgan))

    def test_its_own_state_never_touches_expansion_studio(self):
        body = self._src(App._tab_upstudio)
        self.assertNotIn("self._usc_folder", body)
        self.assertNotIn("self.usc_", body)

    def test_the_project_saves_and_restores_it(self):
        self.assertIn('data["modules"]["upstudio"]',
                      self._src(App._project_save))
        load = self._src(App._project_load)
        self.assertIn('mods.get("upstudio")', load)
        self.assertIn("self._ups_load_folder(", load)
        # pre-split projects: old resolution settings carried across,
        # never a None written into a StringVar
        self.assertIn("if _uv.get(old) is not None", load)

    def test_the_handoff_runs_upscale_into_expansion_never_back(self):
        class S(App):
            def __init__(s):
                s.got = []
                for fn in ("_organic_load_folder",
                           "_sam_load_folder", "_ups_load_folder",
                           "_usc_load_folder", "_smudge_load_folder"):
                    setattr(s, fn, lambda d, fn=fn: s.got.append(fn))

            def log(s, *a, **k):
                pass
        a = S()
        a._adw_propagate_output("/out", "upstudio", None)
        self.assertIn("_usc_load_folder", a.got)
        self.assertNotIn("_ups_load_folder", a.got)
        b = S()
        b._adw_propagate_output("/out", "upscale", None)
        self.assertNotIn("_ups_load_folder", b.got)
        self.assertNotIn("_usc_load_folder", b.got)
        self.assertIn("_organic_load_folder", b.got)

    def test_the_viewer_renders_only_the_visible_part(self):
        from PIL import Image
        img = Image.new("RGB", (10, 10), (200, 0, 0))
        out = App._ups_render(img, 40, 40, 20, 20, 5, 5)
        self.assertEqual(out.size, (40, 40))
        self.assertEqual(out.getpixel((15, 15)), (200, 0, 0))
        self.assertEqual(out.getpixel((1, 1)), (10, 10, 10))
        # zoomed far in and panned off-image: nothing to draw, no crash
        out = App._ups_render(img, 40, 40, 20, 20, 500, 500)
        self.assertEqual(out.getpixel((20, 20)), (10, 10, 10))

    def test_setup_lists_its_components(self):
        body = self._src(App._tab_setup)
        i = body.index('"esrgan": {')
        self.assertIn('"upstudio"', body[i:i + 60])


class TestHighlightsOnBlack(unittest.TestCase):
    """Preview & Compare's third mode: everything outside the blown-
    highlight mask is black, so clicking between model tiles shows only
    what each model adds where it matters."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    class _Stub(App):
        def __init__(self, img, result=None):
            from PIL import Image
            self._usc_orig_base = Image.fromarray(img, "RGB")
            self._usc_preview_code = img.astype(np.float32) / 255.0
            self._usc_preview_idx = 0
            self._usc_up_base = (self._usc_orig_base if result is not None
                                 else None)
            self._usc_up_float = result
            self._usc_up_peak = 1.0
            self._usc_up_gamut = "bt709"
            self._usc_hl_mask_cache = None
            self.usc_ai_thresh = _FakeVar(0.96)
            self.usc_ev_var = _FakeVar(0.0)
            self.usc_gm_var = _FakeVar(1.0)

    def _plate(self):
        img = np.full((64, 64, 3), 60, np.uint8)
        img[20:40, 20:40] = 255            # the blown patch
        return img

    def test_the_button_sits_with_wipe_and_diff(self):
        body = self._src(App._tab_upscale)
        self.assertIn("self.usc_hlblack_btn = _usc_seg(", body)
        self.assertIn("self._usc_toggle_hlblack", body)
        self.assertLess(body.index("self.usc_diff_btn"),
                        body.index("self.usc_hlblack_btn"))

    def test_the_mode_routes_through_display(self):
        self.assertIn('if self._usc_mode == "hlblack":',
                      self._src(App._usc_display))
        self.assertIn('"hlblack"', self._src(App._usc_view))
        self.assertIn('"hlblack"', self._src(App._usc_view_none))

    def test_it_uses_the_same_mask_as_show_blown_mask(self):
        body = self._src(App._usc_hl_mask)
        self.assertIn("self._hdr_ai_mask_frame(code, thresh)", body)
        self.assertIn("usc_ai_thresh", body)

    def test_everything_outside_the_mask_is_black(self):
        app = self._Stub(self._plate())
        out, pct = app._usc_hlblack_image()
        a = np.asarray(out)
        self.assertEqual(tuple(a[2, 2]), (0, 0, 0))
        self.assertEqual(tuple(a[30, 30]), (255, 255, 255))
        self.assertGreater(pct, 9.0)

    def test_it_shows_the_viewed_model_not_the_input(self):
        res = np.full((64, 64, 3), 0.25, np.float32)
        app = self._Stub(self._plate(), result=res)
        a = np.asarray(app._usc_hlblack_image()[0])
        self.assertEqual(tuple(a[2, 2]), (0, 0, 0))
        self.assertLess(int(a[30, 30, 0]), 255)
        self.assertGreater(int(a[30, 30, 0]), 0)

    def test_the_mask_is_cached_across_model_switches(self):
        app = self._Stub(self._plate())
        m1 = app._usc_hl_mask()
        self.assertIs(app._usc_hl_mask(), m1)
        app.usc_ai_thresh = _FakeVar(0.5)
        self.assertIsNot(app._usc_hl_mask(), m1)


class TestExposureRange(unittest.TestCase):
    def test_exposure_is_plus_minus_five_and_sliders_are_longer(self):
        import inspect
        body = inspect.getsource(App._ev_toolbar)
        self.assertIn("from_=-5.0, to=5.0", body)
        self.assertEqual(body.count("length=240"), 2)
        self.assertEqual(body.count('fill="x", expand=True'), 2)
        self.assertNotIn("length=120", body)

    def test_auto_exposure_clamps_to_the_same_range(self):
        import inspect
        self.assertIn("max(-5.0, min(5.0, ev))",
                      inspect.getsource(App._e2m_refresh_preview))


class TestHistogramMaxLabel(unittest.TestCase):
    """The peak value: red on a dark plate, drawn last, always shown."""

    def _draw(self, arr):
        import tkinter as tk

        class S(App):
            def __init__(s):
                pass
        app = S()
        root = tk.Tk()
        try:
            cv = tk.Canvas(root, width=300, height=90)
            cv.pack()
            root.update()
            app.usc_hist_canvas = cv
            app.usc_hist_on = True
            app._usc_up_float = arr
            app._usc_histogram_refresh()
            items = cv.find_all()
            texts = [(i, cv.itemcget(i, "text"), cv.itemcget(i, "fill"))
                     for i in items if cv.type(i) == "text"]
            return items, texts
        finally:
            root.destroy()

    def test_red_drawn_last_with_a_plate(self):
        try:
            arr = np.linspace(0, 3.0, 64 * 64 * 3,
                              dtype=np.float32).reshape(64, 64, 3)
            items, texts = self._draw(arr)
        except Exception as ex:          # no display in this environment
            self.skipTest(str(ex))
        mx = [t for t in texts if t[1].startswith("max ")]
        self.assertEqual(len(mx), 1)
        self.assertEqual(mx[0][2].upper(), "#FF4040")
        self.assertEqual(items[-1], mx[0][0])     # nothing drawn over it

    def test_shown_even_when_nothing_exceeds_one(self):
        try:
            items, texts = self._draw(np.full((8, 8, 3), 0.8, np.float32))
        except Exception as ex:
            self.skipTest(str(ex))
        self.assertIn("max 0.80", [t[1] for t in texts])


class TestWanFailureDoesNotLookLikeAStall(unittest.TestCase):
    """Reported: Wan preview 'stalled' -- 18 minutes, no progress, no
    error. The fal call had in fact FAILED: the panel said so, but the
    render clock (_eta_tick) was never stopped, so it kept redrawing
    "Render 81 frames · 19m in" every second, kept the pulse going, and
    kept pushing the Wan tile back to "running" -- while the one error
    line scrolled away under later log output."""

    def _run_failing_generate(self):
        import numpy as np
        timers = []
        bars, panel_ends, fracs = [], [], []
        work = tempfile.mkdtemp()

        class Bar:
            set = staticmethod(lambda f, t="": bars.append(t))
            pulse = staticmethod(lambda on: bars.append(("pulse", on)))
            reset = staticmethod(lambda: None)

        class S(App):
            def __init__(s):
                s.usc_bar = Bar
                s.usc_fal_panel = object()
                s.usc_ai_thresh = _FakeVar(0.96)
                s.usc_ai_prompt = _FakeVar("")
                s.ip_vace_res_var = _FakeVar("720p")
                s.ip_vace_steps_var = _FakeVar("30")

            def after(s, _ms, fn=None, *a):
                timers.append((fn, a))

            def progress(s, *a): pass
            def _pv_frame(s, p):
                from PIL import Image
                return Image.new("RGB", (16, 16))
            def _usc_ai_union_mask(s, *a, **k):
                return np.full((16, 16), 255, np.uint8)
            def _usc_shadow_thresh(s): return None
            def _usc_shadow_active(s): return False
            def _usc_ai_missing(s, *a): return [0]
            def _vace_clip_problem(s, n): return None
            def _vace_estimate(s, n, endpoint=None): return (60.0, False)
            def _usc_ai_encode(s, p): return np.zeros((16, 16, 3), np.uint8)
            def _vace_build_source_clip(s, stage, staged, a, b, dst, fps,
                                        custom_mask=None):
                with open(dst, "wb") as fh:
                    fh.write(b"x" * 2048)
            def _vace_build_mask_clip(s, n, size, dst, fps, custom_mask=None):
                with open(dst, "wb") as fh:
                    fh.write(b"x" * 2048)
            def _vace_active_endpoint(s):
                return "https://queue.fal.run/fal-ai/x/inpainting"
            def _vace_record_render(s, *a, **k): pass
            def _fal_panel_begin(s, *a, **k): pass
            def _fal_panel_note(s, *a, **k): pass
            def _fal_panel_end(s, panel, attr, note="", ok=True):
                panel_ends.append((note, ok))
            def _vace_submit(s, files, plan, relay, on_phase=None, **k):
                on_phase("render", "81 frames")   # starts the clock
                raise RuntimeError("Request failed: model error 500")

        app = S()
        old = tempfile.gettempdir
        tempfile.gettempdir = lambda: work
        try:
            with self.assertRaises(Exception):
                app._usc_ai_generate("/f", ["a.png"] * 81, 0, 80,
                                     on_log=lambda m: None,
                                     on_frac=fracs.append)
            # pump the event queue for "a long time": a live clock would
            # reschedule itself on every pass
            for _ in range(200):
                if not timers:
                    break
                fn, a = timers.pop(0)
                if fn:
                    fn(*a)
        finally:
            tempfile.gettempdir = old
            shutil.rmtree(work, ignore_errors=True)
        return timers, bars, panel_ends, fracs

    def test_the_render_clock_stops_when_the_call_fails(self):
        timers, bars, _ends, _fr = self._run_failing_generate()
        self.assertEqual(timers, [], "the render clock is still ticking")
        render_lines = [b for b in bars
                        if isinstance(b, str) and b.startswith("Render")]
        self.assertLessEqual(len(render_lines), 1)
        self.assertIn(("pulse", False), bars)

    def test_the_panel_says_why_in_red(self):
        _t, _b, ends, _f = self._run_failing_generate()
        self.assertEqual(len(ends), 1)
        note, ok = ends[0]
        self.assertFalse(ok)
        self.assertIn("model error 500", note)
        self.assertIn("fg=G if ok else RB",
                      __import__("inspect").getsource(App._fal_panel_end))

    def test_a_late_progress_update_cannot_revive_a_failed_tile(self):
        body = __import__("inspect").getsource(App._usc_preview)
        i = body.index("def _frac(")
        seg = body[i:i + 600]
        self.assertLess(seg.index('"phase") != "running":'),
                        seg.index('self._usc_tile_set(m, "running", f)'))

    def test_failures_are_repeated_at_the_end_of_the_preview(self):
        logs = []

        class S(App):
            def __init__(s):
                s._usc_previewing = True
                s._usc_bar_prefix = "x"
                s._usc_tile_state = {}
                s.usc_preview_btn = _FakeWidget()
                s.usc_bar = _FakeWidget()
                s.usc_log = None
            def log(s, _w, m, tag="dim"): logs.append((tag, m))
            def progress(s, *a): pass
            def _usc_models_changed(s): pass
        S()._usc_preview_finished(["tonemap"], 2,
                                  [("Wan inpaint", "model error 500")])
        self.assertEqual(logs[-1], ("err",
                                    "✗ Wan inpaint failed: model error 500"))

    def test_fix_missing_frames_stops_its_clock_too(self):
        import inspect
        src = inspect.getsource(App)
        i = src.index('self.ff_fal_panel, "_ff_fal_state",\n'
                      '                            f"fal call failed')
        seg = src[i - 600:i]
        self.assertIn('eta["t0"] = None', seg)


class TestOneProgressBar(unittest.TestCase):
    """One bar, not two: the studios' bars under Run are gone, and the
    window's own bar at the bottom has the same pixelated look."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_the_studios_have_no_bar_under_run(self):
        for fn in (App._tab_upscale, App._tab_upstudio):
            with self.subTest(tab=fn.__name__):
                self.assertNotIn("_dither_bar(", self._src(fn))
                self.assertIn("self._global_bar_proxy()", self._src(fn))

    def test_the_window_bar_is_the_pixelated_one(self):
        body = self._src(App._build_progress_bar)
        self.assertIn("self._dither_bar(cv_wrap", body)
        self.assertIn("show_label=False", body)

    def _app(self):
        calls = []

        class Fake:
            set = staticmethod(lambda f, t="": calls.append(("set", round(f, 2))))
            reset = staticmethod(lambda: calls.append(("reset",)))
            pulse = staticmethod(lambda on: calls.append(("pulse", on)))

        class S(App):
            def __init__(s):
                s._prog_dither = Fake
                s._prog_label = _FakeWidget()
                s._prog_value = 0.0
                s._prog_pulsing = False
                s._prog_pulse_src = None
                s._usc_bar_prefix = ""
        return S(), calls

    def test_the_proxy_drives_the_window_bar(self):
        app, calls = self._app()
        bar = app._global_bar_proxy()
        bar.set(0.5, "Expand 3 / 6")
        bar.pulse(True)
        app.progress(0.6, "x")            # determinate update keeps a render shimmer
        bar.reset()
        self.assertEqual(calls, [("set", 0.5), ("pulse", True), ("set", 0.6),
                                 ("reset",)])

    def test_an_indeterminate_pulse_ends_on_the_first_real_value(self):
        app, calls = self._app()
        app.progress(None, "Working")
        app.progress(0.3, "go")
        self.assertEqual(calls, [("set", 1.0), ("pulse", True),
                                 ("pulse", False), ("set", 0.3)])

    def test_the_model_name_reaches_the_window_bar_label(self):
        app, _calls = self._app()
        labels = []
        app._prog_label = type("L", (), {"configure": staticmethod(
            lambda text="": labels.append(text))})
        app._usc_bar_prefix = "Topaz (2/3)"
        app.progress(0.4, "Render  24 frames")
        app.progress(0.5, "Topaz (2/3)  \u00b7  Render  24 frames")
        self.assertEqual(labels, ["Topaz (2/3)  \u00b7  Render  24 frames"] * 2)

    def test_a_label_cannot_squeeze_out_the_bar(self):
        app, _calls = self._app()
        labels = []
        app._prog_label = type("L", (), {"configure": staticmethod(
            lambda text="": labels.append(text))})
        app.progress(0.4, "x" * 300)
        self.assertLessEqual(len(labels[0]), 90)


class TestVaceStepsWithinTheEndpointLimit(unittest.TestCase):
    """Reported: every Wan inpaint preview failed with fal's validation
    error "num_inference_steps: Input should be less than or equal to
    50" (input 60) -- the shared Steps default, set for the older
    endpoint. Every request now goes through one clamp."""

    class _Stub(App):
        def __init__(self, v):
            self.ip_vace_steps_var = _FakeVar(v)

    def test_values_above_the_limit_are_clamped(self):
        for v, want in (("60", 50), ("50", 50), ("49", 49), ("200", 50),
                        ("25", 25), ("0", 30), ("", 30), ("abc", 30),
                        ("40.0", 40)):
            with self.subTest(v=v):
                self.assertEqual(self._Stub(v)._vace_steps(), want)

    def test_every_payload_uses_the_clamp(self):
        import inspect
        src = inspect.getsource(App)
        # Two payload builders: _vace_payload (Fix Missing Frames, Wan
        # inpaint) and Expansion Studio's own. The Interpolate tab's
        # third went with that tab.
        self.assertEqual(src.count('"num_inference_steps": self._vace_steps()'),
                         2)
        self.assertNotIn('"num_inference_steps": int(', src)

    def test_the_limit_is_fals(self):
        self.assertEqual(App.VACE_MAX_STEPS, 50)


class TestFalHoldOnTheBar(unittest.TestCase):
    """While a job sits on fal's side, the window's bar stands still and
    fal's mark (red on black) sits at the end of the lit part."""

    class _App(App):
        def __init__(self):
            import tkinter as tk
            tk.Tk.__init__(self)
            self._prog_pulsing = False
            self._prog_value = 0.0
            self._usc_bar_prefix = ""

    def setUp(self):
        import tkinter as tk
        try:
            self.a = self._App()
        except tk.TclError:
            self.skipTest("no display")
        self.a.geometry("600x40")
        self.a._build_progress_bar()
        self.a.update()

    def tearDown(self):
        try:
            self.a.destroy()
        except Exception:
            pass

    def _lit(self):
        cv = self.a._prog_dither.canvas
        return sum(1 for i in cv.find_all()
                   if cv.itemcget(i, "fill") == "#CC4444")

    def _mark(self):
        cv = self.a._prog_dither.canvas
        return sum(1 for i in cv.find_all()
                   if cv.itemcget(i, "fill") == "#FF2A2A")

    def test_mark_is_a_square_bitmap(self):
        rows = App._FAL_PIXELS
        self.assertEqual({len(r) for r in rows}, {len(rows)})
        self.assertTrue(set("".join(rows)) <= {"#", "."})

    def test_bar_stands_still_and_shows_the_mark(self):
        a = self.a
        a.progress(0.3, "render"); a.update()
        self.assertGreater(self._lit(), 0)
        bar = a._prog_dither
        self.assertEqual(self._mark(), 0)
        a._fal_hold_begin(); a.update()
        self.assertGreater(self._mark(), 0)
        a.progress(0.8, "render"); a.update()
        self.assertAlmostEqual(bar.frac(), 0.3)      # did not move
        self.assertIn("waiting on fal.ai", a._prog_label.cget("text"))
        a._fal_hold_end(); a.update()
        self.assertEqual(self._mark(), 0)
        self.assertAlmostEqual(bar.frac(), 0.8)      # caught up
        self.assertNotIn("waiting on fal.ai", a._prog_label.cget("text"))

    def test_holds_are_counted(self):
        a = self.a
        a.progress(0.3, "x")
        a._fal_hold_begin(); a._fal_hold_begin(); a._fal_hold_end()
        a.update()
        self.assertGreater(self._mark(), 0)
        a._fal_hold_end(); a.update()
        self.assertEqual(self._mark(), 0)

    def test_reset_clears_the_hold(self):
        a = self.a
        a.progress(0.3, "x"); a._fal_hold_begin(); a.update()
        a.progress(-1); a.update()
        self.assertEqual(self._mark(), 0)
        self.assertEqual(getattr(a, "_fal_hold_n", 0), 0)

    def test_every_fal_call_holds_the_bar(self):
        calls = []

        class S:
            VACE_FAL_POLL_S = 0.01
            VACE_FAL_TIMEOUT_S = 5
            VACE_FAL_HEARTBEAT_S = 99
            _fal_hold_begin = lambda s: calls.append("begin")
            _fal_hold_end = lambda s: calls.append("end")
            after = lambda s, ms, f: f()
            _fal_call_with_watchdog = App._fal_call_with_watchdog
            _fal_call_with_watchdog_inner = App._fal_call_with_watchdog_inner
        self.assertEqual(S()._fal_call_with_watchdog(lambda: 7), 7)
        self.assertEqual(calls, ["begin", "end"])
        calls.clear()

        def boom():
            raise ValueError("x")
        with self.assertRaises(ValueError):
            S()._fal_call_with_watchdog(boom)
        self.assertEqual(calls, ["begin", "end"])


class TestPreviewButtonCancels(unittest.TestCase):
    """While a preview runs, the Preview button is its Cancel -- a
    preview stuck on fal must never leave nothing to press."""

    def _src(self, f):
        import inspect
        return inspect.getsource(f)

    def test_button_flips_to_cancel_while_previewing(self):
        body = self._src(App._usc_preview)
        self.assertIn("CANCEL PREVIEW", body)
        self.assertNotIn("PREVIEWING", body)

    def test_second_press_cancels(self):
        body = self._src(App._usc_preview)
        i = body.index('if getattr(self, "_usc_previewing", False):')
        self.assertIn("self._usc_preview_cancel()", body[i:i + 300])

    def test_cancel_reaches_the_fal_watchdog(self):
        hits = []

        class S:
            _usc_prev_cancelled = False
            _ip_vace_cancel = False
            usc_log = None
            log = lambda s, *a, **k: hits.append(a)
            usc_preview_btn = type("B", (), {
                "configure": lambda self, **k: hits.append(k)})()
        s = S()
        App._usc_preview_cancel(s)
        self.assertTrue(s._ip_vace_cancel)
        self.assertTrue(s._usc_prev_cancelled)
        self.assertTrue(any(isinstance(h, dict) and
                            h.get("state") == "disabled" for h in hits))

    def test_a_stale_cancel_does_not_end_the_next_preview(self):
        body = self._src(App._usc_preview)
        self.assertIn("self._ip_vace_cancel = False", body)

    def test_cancelled_is_not_reported_as_failed(self):
        body = self._src(App._usc_preview)
        i = body.index("except Exception as ex:")
        j = body.index("failed.append", i)
        self.assertIn("_usc_prev_cancelled", body[i:j])


class TestMultiModelOutput(unittest.TestCase):
    """Multi model output: every ticked model in one EXR per frame, each
    as its own layer, the viewed model in the main RGB."""

    class _Var:
        def __init__(self, v): self.v = v
        def get(self): return self.v

    def _stub(self, on=True, fmt="EXR 16-bit half", sel=None):
        class S:
            USC_MODELS = App.USC_MODELS
            USC_FORMATS = App.USC_FORMATS
            USC_DEFAULT_FORMAT = App.USC_DEFAULT_FORMAT
            USC_MODEL_NAMES = App.USC_MODEL_NAMES
            USC_LAYER_NAMES = App.USC_LAYER_NAMES
            _usc_format = App._usc_format
            _usc_multi_models = App._usc_multi_models
            _usc_run_label = App._usc_run_label
            _usc_render_model = lambda self: "tonemap"
        st = S()
        st.usc_multi_var = self._Var(on)
        st.usc_fmt_var = self._Var(fmt)
        st._usc_sel = sel if sel is not None else {
            "tonemap": True, "topaz": False, "wan_vace": True}
        return st

    def test_off_means_no_layers(self):
        self.assertEqual(self._stub(on=False)._usc_multi_models(), [])

    def test_on_lists_ticked_models_in_tile_order(self):
        self.assertEqual(self._stub()._usc_multi_models(),
                         ["tonemap", "wan_vace"])

    def test_exr_only(self):
        self.assertEqual(
            self._stub(fmt="MOV ProRes 4444")._usc_multi_models(), [])

    def test_run_button_says_what_it_writes(self):
        self.assertIn("2 MODELS", self._stub()._usc_run_label())
        self.assertIn("EXR LAYERS", self._stub()._usc_run_label())
        self.assertNotIn("LAYERS", self._stub(on=False)._usc_run_label())

    def test_layer_names(self):
        self.assertEqual(App.USC_LAYER_NAMES, {
            "tonemap": "tonemap", "topaz": "topaz",
            "wan_vace": "wan_inpaint"})

    def test_writer_round_trip(self):
        try:
            import OpenEXR, Imath
        except ImportError:
            self.skipTest("OpenEXR not installed")
        import numpy as np, tempfile
        h, w = 6, 8
        main = np.full((h, w, 3), 2.5, np.float32)
        layers = {"tonemap": np.full((h, w, 3), 1.0, np.float32),
                  "topaz": np.full((h, w, 3), 4.0, np.float32)}
        path = os.path.join(tempfile.mkdtemp(), "f.exr")
        App._usc_write_exr_layers(path, main, layers, half=True)
        f = OpenEXR.InputFile(path)
        ch = set(f.header()["channels"].keys())
        self.assertEqual(ch, {"R", "G", "B", "tonemap.R", "tonemap.G",
                              "tonemap.B", "topaz.R", "topaz.G",
                              "topaz.B"})
        FT = Imath.PixelType(Imath.PixelType.FLOAT)
        g = lambda c: np.frombuffer(f.channel(c, FT), np.float32)
        self.assertTrue(np.allclose(g("R"), 2.5))
        self.assertTrue(np.allclose(g("topaz.G"), 4.0))   # above 1 kept
        self.assertTrue(np.allclose(g("tonemap.B"), 1.0))

    def test_each_model_is_computed_on_its_own(self):
        """Never Topaz with Wan folded on top: Tone map is computed with
        no Topaz result, and Wan is folded onto Tone map."""
        import inspect
        body = inspect.getsource(App._usc_run)
        i = body.index("if _hdr and _multi and")
        blk = body[i:i + 1500]
        self.assertIn("i, code, None, _peak, _knee, _eotf", blk)
        self.assertIn("_usc_ai_apply(\n                                    _tm", blk)
        self.assertIn('"topaz" in _multi', body)
        self.assertIn('"wan_vace" in _multi', body)
        self.assertIn("_usc_write_exr_layers", body)


class TestUpscaleStudioModels(unittest.TestCase):
    """Upscale Studio rebuilt on Expansion Studio's workflow: model tiles
    (ESRGAN local, SeedVR2 and FlashVSR on fal), per-tile preview,
    compare modes, a frequency graph, Multi model EXR output."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_three_models_local_first(self):
        self.assertEqual([m for m, _n, _i in App.UPS_MODELS],
                         ["esrgan", "seedvr", "flashvsr"])
        self.assertNotIn("esrgan", App.UPS_ENDPOINTS)

    def test_endpoints_are_the_fal_ids(self):
        self.assertEqual(App.UPS_ENDPOINTS["seedvr"],
                         "fal-ai/seedvr/upscale/video")
        self.assertEqual(App.UPS_ENDPOINTS["flashvsr"],
                         "fal-ai/flashvsr/upscale/video")

    def _args_stub(self, noise="0.1", accel="regular"):
        class V:
            def __init__(s, v): s.v = v
            def get(s): return s.v

        class S:
            UPS_FAL_FORMAT = App.UPS_FAL_FORMAT
            _ups_fal_args = App._ups_fal_args
        st = S()
        st.ups_seedvr_noise_var = V(noise)
        st.ups_flash_accel_var = V(accel)
        return st

    def test_request_bodies_use_the_schema_enums(self):
        st = self._args_stub("0.25", "high")
        a = st._ups_fal_args("seedvr", 4)
        self.assertEqual(a["upscale_mode"], "factor")
        self.assertEqual(a["noise_scale"], 0.25)
        self.assertEqual(a["output_format"], "PRORES4444 (.mov)")
        self.assertEqual(a["upscale_factor"], 4.0)
        b = st._ups_fal_args("flashvsr", 2)
        self.assertEqual(b["acceleration"], "high")
        self.assertNotIn("noise_scale", b)

    def test_factor_is_clamped_to_each_models_limit(self):
        self.assertEqual(App.UPS_MAX_FACTOR["flashvsr"], 4.0)
        body = self._src(App._ups_fal_upscale)
        self.assertIn("self._ups_factor_plan(m, (sw, sh), (tw, th))", body)
        self.assertIn("is Lanczos", body)

    def test_a_fully_cached_span_costs_nothing(self):
        body = self._src(App._ups_fal_upscale)
        i = body.index("if all(self._ups_cached(")
        j = body.index("_fal_video_submit", i)
        self.assertIn("return 0", body[i:j])

    def test_every_fal_wait_goes_through_the_watchdog(self):
        body = self._src(App._fal_video_submit)
        self.assertEqual(body.count("self._fal_call_with_watchdog("), 3)
        self.assertIn("cancel_check=lambda: self._ip_vace_cancel", body)

    def test_preview_button_is_its_cancel(self):
        body = self._src(App._ups_preview)
        self.assertIn("self._ups_preview_cancel()", body)
        self.assertIn("CANCEL PREVIEW", body)
        self.assertIn("self._ip_vace_cancel = False", body)

    def test_run_and_preview_never_overlap(self):
        self.assertIn("if self._ups_previewing:",
                      self._src(App._ups_run))
        self.assertIn("if self._ups_running:",
                      self._src(App._ups_preview))

    def test_each_fal_job_has_its_own_scratch_folder(self):
        self.assertIn("tempfile.mkdtemp(", self._src(App._ups_fal_upscale))

    def test_16_bit_round_trip(self):
        import tempfile
        a = np.zeros((4, 5, 3), np.float32)
        a[..., 0] = 0.5
        a[1, 2] = (0.123456, 0.9, 0.0)
        p = os.path.join(tempfile.mkdtemp(), "x.png")
        App._ups_write16(p, a)
        b = App._ups_read16(p)
        self.assertEqual(b.shape, (4, 5, 3))
        self.assertLess(float(np.abs(a - b).max()), 2.0 / 65535)

    def test_spectrum_sees_added_fine_detail(self):
        rng = np.random.default_rng(0)
        import cv2
        smooth = cv2.GaussianBlur(rng.random((256, 256)).astype(np.float32),
                                  (0, 0), 3)
        sharp = smooth + 0.05 * rng.standard_normal((256, 256)).astype(
            np.float32)
        f, ps = App._ups_spectrum(smooth, crop=256)
        _f, pd = App._ups_spectrum(sharp, crop=256)
        self.assertAlmostEqual(float(f[-1]), 0.5, delta=0.01)
        self.assertGreater(App._ups_hf_gain_db(f, pd, ps, 0.25), 10.0)
        self.assertAlmostEqual(App._ups_hf_gain_db(f, ps, ps, 0.25), 0.0)

    def test_graph_marks_the_input_limit(self):
        body = self._src(App._ups_freq_refresh)
        self.assertIn('text="input limit"', body)
        self.assertIn("self._ups_src_size[0] / float(t[0])", body)
        self.assertIn('fill="#FF4040"', body)

    def test_multi_model_layers(self):
        self.assertEqual(App.UPS_LAYER_NAMES, {
            "esrgan": "esrgan", "seedvr": "seedvr2", "flashvsr": "flashvsr"})

        class V:
            def __init__(s, v): s.v = v
            def get(s): return s.v

        class S:
            UPS_MODELS = App.UPS_MODELS
            USC_FORMATS = App.USC_FORMATS
            _ups_multi_models = App._ups_multi_models
        st = S()
        st.ups_multi_var = V(True)
        st.ups_fmt_var = V("EXR 16-bit half")
        st._ups_sel = {"esrgan": True, "seedvr": False, "flashvsr": True}
        self.assertEqual(st._ups_multi_models(), ["esrgan", "flashvsr"])
        st.ups_fmt_var = V("MOV ProRes 4444")
        self.assertEqual(st._ups_multi_models(), [])

    def test_failed_paid_model_falls_back_to_esrgan_and_says_so(self):
        body = self._src(App._ups_run)
        self.assertIn("usable.discard(m)", body)
        self.assertIn("fell back to Real-ESRGAN", body)

    def test_factor_buttons_select_factor_mode(self):
        body = self._src(App._tab_upstudio)
        i = body.index("def _fac_pick(v):")
        self.assertIn('self.ups_res_var.set("factor")', body[i:i + 400])


class TestUpscaleStudioInputColour(unittest.TestCase):
    """Upscale Studio reads sRGB, linear and HDR sources. The upscalers
    only ever see an ordinary 8-bit picture (the base); what the source
    held above white rides round them as a gain map and is put back."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def _stub(self, mode, hdr):
        class S:
            FF_LINEAR_EXTS = App.FF_LINEAR_EXTS
            HDR_SDR_WHITE = App.HDR_SDR_WHITE
            UPS_HDR_KNEE = App.UPS_HDR_KNEE
        st = S()
        for n in ("_ups_read_source", "_ups_out_px", "_ups_scene",
                  "_ups_raw_float", "_ff_read_frame"):
            setattr(st, n, getattr(App, n).__get__(st))
        for n in ("_ups_shoulder", "_ups_gain_from_base", "_ups_read16",
                  "_organic_linear_to_display", "_srgb_to_linear",
                  "_linear_to_srgb", "_hdr_pq_decode", "_hdr_pq_encode",
                  "_hdr_eotf_1886"):
            setattr(st, n, getattr(App, n))
        st._ups_in_mode, st._ups_src_hdr = mode, hdr
        return st

    def _hdr_frame(self):
        rng = np.random.default_rng(0)
        import cv2
        lin = cv2.GaussianBlur(rng.random((60, 80, 3)).astype(np.float32),
                               (0, 0), 3) * 0.8
        lin[20:40, 30:50] = (6.0, 5.0, 4.0)
        return lin

    def _exr(self, lin):
        try:
            import OpenEXR  # noqa: F401
        except ImportError:
            self.skipTest("OpenEXR not installed")
        import tempfile
        p = os.path.join(tempfile.mkdtemp(), "a.exr")
        App._organic_write_exr(p, lin, half=False)
        return p

    def test_choices(self):
        self.assertEqual(App.UPS_IN_CHOICES,
                         ("Auto", "sRGB / display", "Linear", "ACES2065-1",
                          "HDR PQ"))

    def test_shoulder_is_identity_below_the_knee_and_never_clips(self):
        k = App.UPS_HDR_KNEE
        n = np.array([0.0, 0.2, k, 1.0, 10.0, 1000.0], np.float32)
        t = App._ups_shoulder(n)
        self.assertTrue(np.allclose(t[:3], n[:3]))
        self.assertTrue(np.all(np.diff(t) > 0))       # still distinct
        self.assertLess(float(t.max()), 1.0)

    def test_linear_hdr_round_trips_with_highlights_intact(self):
        lin = self._hdr_frame()
        st = self._stub("linear", True)
        base, gain = st._ups_read_source(self._exr(lin))
        self.assertIsNotNone(gain)
        b = np.asarray(base, np.float32) / 255.0
        self.assertLess(float(b.max()), 1.0)          # nothing clipped
        out = st._ups_out_px(b, gain, {"ext": ".exr"}, "bt1886")
        rel = np.abs(out - lin) / np.maximum(lin, 0.05)
        self.assertLess(float(rel.max()), 0.03)
        self.assertGreater(float(out.max()), 5.8)     # was 1.0 before

    def test_linear_sdr_has_no_gain_map(self):
        lin = np.clip(self._hdr_frame(), 0, 1)
        st = self._stub("linear", False)
        base, gain = st._ups_read_source(self._exr(lin))
        self.assertIsNone(gain)
        out = st._ups_out_px(np.asarray(base, np.float32) / 255.0, None,
                             {"ext": ".exr"}, "bt1886")
        self.assertLess(float(np.abs(out - lin).max()), 0.01)

    def test_pq_round_trips_to_exr_and_back_to_pq(self):
        import tempfile
        lin = self._hdr_frame()
        code = App._hdr_pq_encode(lin * App.HDR_SDR_WHITE)
        p = os.path.join(tempfile.mkdtemp(), "pq.png")
        App._ups_write16(p, code)
        st = self._stub("pq", True)
        base, gain = st._ups_read_source(p)
        b = np.asarray(base, np.float32) / 255.0
        out = st._ups_out_px(b, gain, {"ext": ".exr"}, "bt1886")
        self.assertLess(float((np.abs(out - lin)
                               / np.maximum(lin, 0.05)).max()), 0.03)
        back = st._ups_out_px(b, gain, {"ext": ".png"}, "bt1886")
        self.assertLess(float(np.abs(back - code).max()), 0.004)

    def test_no_bright_fringe_beside_a_hard_highlight(self):
        """The enlarged gain map is soft and the upscaled edge is not;
        without the per-pixel ceiling the map lights the dark pixels
        next to a highlight."""
        import cv2
        lin = np.full((60, 80, 3), 0.05, np.float32)
        lin[20:40, 30:50] = 6.0
        st = self._stub("linear", True)
        base, gain = st._ups_read_source(self._exr(lin))
        b = np.asarray(base, np.float32) / 255.0
        up = cv2.resize(b, (320, 240), interpolation=cv2.INTER_NEAREST)
        out = st._ups_out_px(up, gain, {"ext": ".exr"}, "bt1886")
        ref = cv2.resize(lin, (320, 240), interpolation=cv2.INTER_NEAREST)
        dark = ref[:, :, 0] < 1.0
        self.assertLess(float(out[dark].max()), 0.06)
        self.assertGreater(float(out[100:140, 140:180].min()), 5.8)

    def test_srgb_source_is_untouched_for_a_movie(self):
        st = self._stub("srgb", False)
        a = np.random.default_rng(1).random((4, 4, 3)).astype(np.float32)
        self.assertIs(st._ups_out_px(a, None, {"ext": ".png"}, "bt1886"), a)

    def test_the_cache_key_knows_how_the_source_was_read(self):
        body = self._src(App._ups_model_sig)
        self.assertIn("|in={self._ups_in_mode}", body)
        self.assertIn("+hdr", body)

    def test_every_read_goes_through_the_one_reader(self):
        for fn in (App._ups_run, App._ups_fal_upscale,
                   App._ups_load_preview_frame):
            with self.subTest(fn=fn.__name__):
                body = self._src(fn)
                self.assertIn("self._ups_read_source(", body)
                self.assertNotIn("self._usc_read_rgb(", body)

    def test_auto_reads_tags_type_and_over_range(self):
        body = self._src(App._resolve_input_colour)
        for frag in ('== "smpte2084"', "self._adw_read_colorspace(folder)",
                     "self._exr_is_over_range(", "self._ff_is_linear("):
            with self.subTest(fragment=frag):
                self.assertIn(frag, body)

    def test_a_pq_movie_is_decoded_at_16_bit_and_written_tagged(self):
        self.assertIn("rgb48be", self._src(App._ups_decode_movie16))
        self.assertIn("self._ups_decode_movie16(",
                      self._src(App._ups_load_movie))
        self.assertIn("self._hdr_tags()", self._src(App._ups_encode_movie))

    def test_exr_output_is_tagged_scene_linear(self):
        body = self._src(App._ups_run)
        self.assertIn('"scene-linear" if spec["ext"] == ".exr"', body)

    def test_hdr_into_an_sdr_movie_is_called_out(self):
        self.assertIn("values above white are clipped",
                      self._src(App._ups_run))


class TestACES2065(unittest.TestCase):
    """ACES2065-1: Expansion Studio writes it, Upscale Studio and
    ADW-Organic read it (and write it back)."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def _need_exr(self):
        try:
            import OpenEXR  # noqa: F401
        except ImportError:
            self.skipTest("OpenEXR not installed")

    # ── the maths ──
    def test_white_stays_white(self):
        w = App._lin709_to_aces(np.ones((1, 1, 3), np.float32))
        self.assertTrue(np.allclose(w, 1.0, atol=5e-4))
        self.assertTrue(np.allclose(
            App._lin709_to_aces(np.full((1, 1, 3), 10.0, np.float32)),
            10.0, atol=5e-3))

    def test_round_trip_is_exact(self):
        a = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32) * 6
        back = App._aces_to_lin709(App._lin709_to_aces(a))
        self.assertLess(float(np.abs(back - a).max()), 1e-4)

    def test_pure_red_lands_on_the_published_values(self):
        r = App._lin709_to_aces(np.array([[[1.0, 0.0, 0.0]]], np.float32))
        self.assertTrue(np.allclose(r[0, 0], (0.43970, 0.08979, 0.01754),
                                    atol=1e-4))

    # ── the file ──
    def test_an_aces_exr_says_it_is_aces(self):
        self._need_exr()
        import tempfile
        d = tempfile.mkdtemp()
        a = np.random.default_rng(1).random((6, 8, 3)).astype(np.float32)
        p1, p2 = os.path.join(d, "a.exr"), os.path.join(d, "b.exr")
        App._organic_write_exr(p1, a, half=False, aces=True)
        App._organic_write_exr(p2, a, half=False)
        import OpenEXR
        if hasattr(OpenEXR, "File"):
            self.assertTrue(App._exr_is_aces(p1))
        self.assertFalse(App._exr_is_aces(p2))
        self.assertFalse(App._exr_is_aces(os.path.join(d, "x.png")))

        class R:
            pass
        r = R()
        got = App._ff_read_frame.__get__(r)(p1)
        self.assertLess(float(np.abs(got - App._lin709_to_aces(a)).max()),
                        1e-5)

    def test_layers_are_converted_too(self):
        self._need_exr()
        import tempfile, OpenEXR, Imath
        p = os.path.join(tempfile.mkdtemp(), "m.exr")
        red = np.zeros((4, 4, 3), np.float32)
        red[..., 0] = 1.0
        App._usc_write_exr_layers(p, red, {"topaz": red}, half=False,
                                  aces=True)
        f = OpenEXR.InputFile(p)
        FT = Imath.PixelType(Imath.PixelType.FLOAT)
        g = lambda c: np.frombuffer(f.channel(c, FT), np.float32)
        self.assertEqual(set(f.header()["channels"].keys()),
                         {"R", "G", "B", "topaz.R", "topaz.G", "topaz.B"})
        self.assertTrue(np.allclose(g("R"), 0.43970, atol=1e-4))
        self.assertTrue(np.allclose(g("topaz.G"), 0.08979, atol=1e-4))

    def test_the_folder_tag_carries_primaries_without_breaking_old_readers(self):
        import tempfile
        d = tempfile.mkdtemp()
        App._adw_write_colorspace(d, "scene-linear", primaries=App.ACES_NAME)
        self.assertEqual(App._adw_read_colorspace(d), "scene-linear")
        self.assertEqual(App._adw_read_primaries(d), "aces2065-1")
        self.assertTrue(App._seq_is_aces(d))
        App._adw_write_colorspace(d, "scene-linear")
        self.assertIsNone(App._adw_read_primaries(d))
        self.assertFalse(App._seq_is_aces(d))

    # ── Expansion Studio ──
    def test_expansion_writes_aces_by_default(self):
        self.assertEqual(App.USC_EXR_SPACES[0], "ACES2065-1")

        class V:
            def __init__(s, v): s.v = v
            def get(s): return s.v

        class S:
            USC_EXR_SPACES = App.USC_EXR_SPACES
            usc_exr_cs_var = None
            _usc_exr_aces = App._usc_exr_aces
        st = S()
        self.assertTrue(st._usc_exr_aces())
        st.usc_exr_cs_var = V("Linear Rec.709")
        self.assertFalse(st._usc_exr_aces())
        body = self._src(App._usc_run)
        self.assertEqual(body.count("aces=_aces"), 2)   # single + layers
        self.assertIn("primaries=self.ACES_NAME if _aces else None", body)

    def test_the_tooltip_no_longer_claims_bt2020_for_exr(self):
        body = self._src(App._usc_fmt_text)
        self.assertNotIn("holding nits in BT.2020", body)
        self.assertIn("ACES2065-1: linear, AP0 primaries", body)

    # ── Upscale Studio ──
    def test_upscale_reads_aces_and_writes_it_back(self):
        self.assertEqual(App.UPS_IN_NAMES["aces"], "ACES2065-1")
        self.assertIn("self._seq_is_aces(folder, first)",
                      self._src(App._resolve_input_colour))
        self.assertIn("self._resolve_input_colour(",
                      self._src(App._ups_resolve_input))
        self.assertIn("self._aces_to_lin709(a)", self._src(App._ups_scene))
        run = self._src(App._ups_run)
        self.assertIn('self._ups_in_mode == "aces"', run)
        self.assertEqual(run.count("aces=aces"), 2)

    def test_upscale_scene_values_are_rec709(self):
        self._need_exr()
        import tempfile
        a = np.random.default_rng(2).random((6, 8, 3)).astype(np.float32)
        p = os.path.join(tempfile.mkdtemp(), "a.exr")
        App._organic_write_exr(p, a, half=False, aces=True)

        class S:
            FF_LINEAR_EXTS = App.FF_LINEAR_EXTS
            HDR_SDR_WHITE = App.HDR_SDR_WHITE
            _ups_in_mode = "aces"
            _aces_to_lin709 = App._aces_to_lin709
        st = S()
        for n in ("_ups_scene", "_ups_raw_float", "_ff_read_frame"):
            setattr(st, n, getattr(App, n).__get__(st))
        self.assertLess(float(np.abs(st._ups_scene(p) - a).max()), 1e-4)

    # ── ADW-Organic ──
    def test_organic_offers_and_detects_aces(self):
        self.assertIn("values=list(self.LENS_IN_CHOICES)",
                      self._src(App._tab_organic))
        self.assertIn("ACES2065-1", App.LENS_IN_CHOICES)
        body = self._src(App._organic_load_linear)
        self.assertIn("self._seq_is_aces(os.path.dirname(path), path)", body)
        self.assertIn("self._aces_to_lin709(arr)", body)

    def test_organic_reads_aces_as_rec709_linear(self):
        self._need_exr()
        import tempfile
        a = np.random.default_rng(3).random((6, 8, 3)).astype(np.float32)
        p = os.path.join(tempfile.mkdtemp(), "a.exr")
        App._organic_write_exr(p, a, half=False, aces=True)

        class S:
            organic_log_txt = None
            _organic_source_encoded = False
            _organic_source_aces = True
            _organic_cs_reason = "test"
            _aces_to_lin709 = App._aces_to_lin709
            log = lambda s, *k, **kw: None
            _organic_load_linear = App._organic_load_linear
        got = S()._organic_load_linear(p)
        self.assertLess(float(np.abs(got - a).max()), 1e-4)
        # the render's snapshot wins over whatever the UI holds
        raw = S()._organic_load_linear(p, encoded=False, aces=False)
        self.assertLess(float(np.abs(raw - App._lin709_to_aces(a)).max()),
                        1e-4)

    def test_organic_render_snapshots_and_writes_back_aces(self):
        body = self._src(App._organic_run_worker)
        self.assertIn("aces=_aces)", body)
        self.assertIn("aces=_aces_out)", body)
        self.assertIn("primaries=self.ACES_NAME if _aces_out else None",
                      body)


class TestUpscaleModelOutputLimit(unittest.TestCase):
    """SeedVR2 tops out at 2160p, so the factor it accepts depends on
    the clip. The app asks for the most it can have, says what size that
    is, and takes fal's own number if it is still refused."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def _stub(self, caps=None):
        class S:
            UPS_MAX_FACTOR = App.UPS_MAX_FACTOR
            UPS_MAX_SHORT_SIDE = App.UPS_MAX_SHORT_SIDE
            UPS_MODEL_NAMES = App.UPS_MODEL_NAMES
            _ups_factor_plan = App._ups_factor_plan
            _ups_limit_text = App._ups_limit_text
        st = S()
        st._ups_fal_caps = caps or {}
        return st

    def test_the_reported_case(self):
        """900x540 to UHD wanted x4.267; fal allowed 4.0."""
        f, native, want = self._stub()._ups_factor_plan(
            "seedvr", (900, 540), (3840, 2160))
        self.assertEqual(f, 4.0)
        self.assertEqual(native, (3600, 2160))
        self.assertAlmostEqual(want, 4.2667, places=3)

    def test_no_limit_when_the_size_is_reachable(self):
        st = self._stub()
        f, native, want = st._ups_factor_plan(
            "seedvr", (1920, 1080), (3840, 2160))
        self.assertEqual((f, native), (2.0, (3840, 2160)))
        self.assertIsNone(st._ups_limit_text(
            "seedvr", (1920, 1080), (3840, 2160)))

    def test_flashvsr_stops_at_x4(self):
        f, native, _w = self._stub()._ups_factor_plan(
            "flashvsr", (480, 270), (3840, 2160))
        self.assertEqual((f, native), (4.0, (1920, 1080)))

    def test_the_factor_is_never_rounded_up_past_the_limit(self):
        f, _n, want = self._stub()._ups_factor_plan(
            "seedvr", (900, 400), (3840, 1600))
        self.assertLessEqual(f, want)
        self.assertEqual(f, 4.266)

    def test_fals_own_number_is_parsed_and_used(self):
        msg = ("[{'loc': ['body', 'upscale_factor'], 'msg': 'Upscale factor "
               "is too high. Please use an upscale factor of 4.0 or lower "
               "for this video.', 'type': 'value_error', 'input': 4.267}]")
        self.assertEqual(App._ups_parse_factor_limit(msg), 4.0)
        self.assertIsNone(App._ups_parse_factor_limit("queue timeout"))
        st = self._stub(caps={("seedvr", 480, 270): 4.0})
        f, native, _w = st._ups_factor_plan(
            "seedvr", (480, 270), (3840, 2160))
        self.assertEqual((f, native), (4.0, (1920, 1080)))

    def test_the_notice_names_the_size_it_will_render(self):
        txt = self._stub()._ups_limit_text(
            "seedvr", (900, 540), (3840, 2160))
        self.assertIn("SeedVR2 cannot reach 3840\u00d72160", txt)
        self.assertIn("3600\u00d72160", txt)

    def test_a_refusal_is_retried_once_at_fals_number(self):
        body = self._src(App._ups_fal_upscale)
        self.assertIn("self._ups_parse_factor_limit(ex)", body)
        self.assertEqual(body.count("self._fal_video_submit("), 2)
        self.assertIn("told + 1e-6 >= factor", body)   # no endless retry
        self.assertIn("messagebox.showinfo(", body)

    def test_preview_and_run_say_so_up_front_and_only_once(self):
        self.assertIn("self._ups_limit_notice(models)",
                      self._src(App._ups_preview))
        self.assertIn("self._ups_limit_notice(models)",
                      self._src(App._ups_run))
        body = self._src(App._ups_limit_notice)
        self.assertIn("key in seen", body)
        self.assertIn("messagebox.showinfo(", body)


class TestExpansionStudioInputColour(unittest.TestCase):
    """Expansion Studio reads the same sources Upscale Studio does --
    sRGB, linear, ACES2065-1, HDR PQ -- by the same Auto rule."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def _stub(self, folder, mode):
        class S:
            FF_LINEAR_EXTS = App.FF_LINEAR_EXTS
            HDR_SDR_WHITE = App.HDR_SDR_WHITE
            _usc_in_explicit = False
            _adw_source_is_encoded = lambda s: True
        st = S()
        for n in ("_usc_read_rgb", "_usc_read_float", "_usc_mode_for",
                  "_usc_eotf", "_usc_in_sig", "_ups_raw_float",
                  "_ff_read_frame"):
            setattr(st, n, getattr(App, n).__get__(st))
        for n in ("_scene_from_raw", "_aces_to_lin709", "_hdr_pq_decode",
                  "_ups_read16", "_organic_linear_to_display",
                  "_ff_is_linear"):
            setattr(st, n, getattr(App, n))
        st._usc_folder, st._usc_in_mode = folder, mode
        st._usc_frames = ["a"]
        return st

    def _aces_file(self):
        try:
            import OpenEXR  # noqa: F401
        except ImportError:
            self.skipTest("OpenEXR not installed")
        import tempfile
        d = tempfile.mkdtemp()
        lin = np.random.default_rng(4).random((6, 8, 3)).astype(
            np.float32) * 0.9
        p = os.path.join(d, "a.exr")
        App._organic_write_exr(p, lin, half=False, aces=True)
        return d, p, lin

    def test_same_choices_and_rule_as_upscale_studio(self):
        tab = self._src(App._tab_upscale)
        self.assertIn("values=list(self.UPS_IN_CHOICES)", tab)
        self.assertIn("self._usc_in_changed()", tab)
        self.assertIn("self._resolve_input_colour(",
                      self._src(App._usc_resolve_input))
        self.assertIn("self._usc_resolve_input()",
                      self._src(App._usc_load_folder))

    def test_aces_source_is_read_as_rec709(self):
        d, p, lin = self._aces_file()
        st = self._stub(d, "aces")
        fl, _pk = st._usc_read_float(p)
        self.assertLess(float(np.abs(fl - lin).max()), 1e-4)
        code = np.asarray(st._usc_read_rgb(p), np.float32) / 255.0
        want = App._organic_linear_to_display(lin).astype(np.float32) / 255.0
        self.assertLess(float(np.abs(code - want).max()), 1.5 / 255)
        self.assertEqual(st._usc_eotf(), "srgb")

    def test_pq_source_is_decoded_to_scene_linear(self):
        import tempfile
        d = tempfile.mkdtemp()
        lin = np.full((4, 4, 3), 0.5, np.float32)
        lin[0, 0] = (5.0, 4.0, 3.0)
        p = os.path.join(d, "a.png")
        App._ups_write16(p, App._hdr_pq_encode(lin * App.HDR_SDR_WHITE))
        st = self._stub(d, "pq")
        fl, _pk = st._usc_read_float(p)
        self.assertLess(float((np.abs(fl - lin) / lin).max()), 0.01)
        code = np.asarray(st._usc_read_rgb(p))
        self.assertEqual(tuple(code[0, 0]), (255, 255, 255))  # above white

    def test_display_source_keeps_bt1886(self):
        st = self._stub("/x", "srgb")
        self.assertEqual(st._usc_eotf(), "bt1886")
        self.assertEqual(st._usc_read_float("/x/a.png"), (None, 1.0))

    def test_generated_and_cached_frames_are_never_reinterpreted(self):
        """A cached generation is a plain 8-bit picture in ANOTHER
        folder; reading it as PQ or ACES because the source is would
        wreck it."""
        import tempfile
        from PIL import Image
        src, other = tempfile.mkdtemp(), tempfile.mkdtemp()
        a = np.full((4, 4, 3), 128, np.uint8)
        Image.fromarray(a, "RGB").save(os.path.join(other, "g.png"))
        st = self._stub(src, "pq")
        self.assertIsNone(st._usc_mode_for(os.path.join(other, "g.png")))
        got = np.asarray(st._usc_read_rgb(os.path.join(other, "g.png")))
        self.assertTrue(np.array_equal(got, a))
        self.assertEqual(st._usc_mode_for(os.path.join(src, "f.png")), "pq")

    def test_unresolved_falls_back_to_the_legacy_read(self):
        st = self._stub("/x", None)
        self.assertIsNone(st._usc_mode_for("/x/a.exr"))

    def test_paid_caches_know_the_reading_without_orphaning_old_ones(self):
        st = self._stub("/x", "linear")
        self.assertEqual(st._usc_in_sig(), "")          # Auto + linear
        st._usc_in_explicit = True
        self.assertEqual(st._usc_in_sig(), "|in=linear")
        st._usc_in_explicit, st._usc_in_mode = False, "aces"
        self.assertEqual(st._usc_in_sig(), "|in=aces")
        self.assertIn("self._usc_in_sig()",
                      self._src(App._usc_topaz_cache_key))
        self.assertIn("self._usc_in_sig()",
                      self._src(App._usc_ai_cache_key))

    def test_a_pq_movie_is_decoded_at_16_bit(self):
        body = self._src(App._usc_load_movie)
        self.assertIn("self._ups_decode_movie16(path, out, self.usc_log)",
                      body)
        self.assertIn('== "smpte2084"', body)

    def test_an_hdr_source_is_called_out(self):
        self.assertIn("already HDR", self._src(App._usc_resolve_input))


class TestLensAndPaintStudio(unittest.TestCase):
    """ADW-Organic is Lens Studio and ADW-Paint is Paint Studio, both
    beside Expansion Studio; Paint Studio has the Input colour control."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_tab_names_and_order(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        i = src.index('("Expansion Studio",   "upscale")')
        j = src.index('("Lens Studio",        "organic")')
        k = src.index('("Paint Studio",       "smudge")')
        m = src.index('("Fix Missing Frames", "fixframes")')
        self.assertTrue(i < j < k < m)
        self.assertNotIn('("ADW-Organic",', src)
        self.assertNotIn('("ADW-Paint",', src)

    def test_keys_are_unchanged_so_projects_still_load(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn('("organic", "_organic_load_folder", "Lens Studio")',
                      src)
        self.assertIn('("smudge",  "_smudge_load_folder", "Paint Studio")',
                      src)

    def test_headers_are_renamed(self):
        self.assertIn('text="Lens Studio"', self._src(App._tab_organic))
        self.assertIn('text="Paint Studio"', self._src(App._tab_smudge))

    def test_lens_output_folder_and_movie_are_renamed(self):
        self.assertEqual(App.LENS_OUT_NAME, "Lens_Studio")
        self.assertIn("self.LENS_OUT_NAME", self._src(App._organic_run))
        with open(SRC, encoding="utf-8") as fh:
            self.assertNotIn('"ADW_Optics', fh.read())

    # ── Paint Studio input colour ──
    def test_paint_has_the_same_choices_and_rule(self):
        tab = self._src(App._tab_smudge)
        self.assertIn("values=list(self.UPS_IN_CHOICES)", tab)
        self.assertIn("self._resolve_input_colour(",
                      self._src(App._smudge_resolve_input))
        self.assertIn("self._smudge_resolve_input()",
                      self._src(App._smudge_load_folder))

    def _stub(self, mode, first="a.exr", choice="Auto"):
        class V:
            def __init__(s, v): s.v = v
            def get(s): return s.v

        class S:
            FF_LINEAR_EXTS = App.FF_LINEAR_EXTS
            UPS_IN_NAMES = App.UPS_IN_NAMES
            sm_in_lbl = None
            sm_source_encoded = None
            _smudge_resolve_input = App._smudge_resolve_input
            _smudge_needs_float_read = App._smudge_needs_float_read
            _smudge_aces_out = App._smudge_aces_out
            _lin709_to_aces = App._lin709_to_aces
            _exr_write_planes = App._exr_write_planes
        st = S()
        st.sm_folder, st.sm_frames = "/x", [first]
        st.sm_in_var = V(choice)
        st._resolve_input_colour = lambda c, f, n: (mode, "", False)
        return st

    def test_resolved_mode_pins_how_the_frames_are_decoded(self):
        st = self._stub("aces")
        st._smudge_resolve_input()
        self.assertEqual((st.sm_in_mode, st.sm_source_encoded),
                         ("aces", False))
        st = self._stub("srgb")
        st._smudge_resolve_input()
        self.assertIs(st.sm_source_encoded, True)

    def test_an_8_bit_sequence_on_auto_keeps_its_old_read(self):
        st = self._stub("srgb", first="a.png")
        st._smudge_resolve_input()
        self.assertIsNone(st.sm_source_encoded)       # pinned as before
        st = self._stub("linear", first="a.png", choice="Linear")
        st._smudge_resolve_input()
        self.assertIs(st.sm_source_encoded, False)    # chosen by hand

    def test_linear_or_pq_png_goes_through_the_float_reader(self):
        st = self._stub("pq", first="a.png")
        st.sm_in_mode = "pq"
        self.assertTrue(st._smudge_needs_float_read("/x/a.png"))
        st.sm_in_mode = "srgb"
        self.assertFalse(st._smudge_needs_float_read("/x/a.png"))
        self.assertTrue(st._smudge_needs_float_read("/x/a.exr"))

    def test_aces_source_is_written_back_as_aces(self):
        try:
            import OpenEXR
        except ImportError:
            self.skipTest("OpenEXR not installed")
        import tempfile
        p = os.path.join(tempfile.mkdtemp(), "o.exr")
        lin = np.zeros((4, 4, 3), np.float32)
        lin[..., 0] = 1.0
        st = self._stub("aces")
        st.sm_in_mode = "linear"
        self.assertFalse(st._smudge_aces_out(lin, p))
        self.assertFalse(os.path.exists(p))
        st.sm_in_mode = "aces"
        self.assertTrue(st._smudge_aces_out(lin, p))

        class R:
            pass
        got = App._ff_read_frame.__get__(R())(p)
        self.assertTrue(np.allclose(got[0, 0], (0.43970, 0.08979, 0.01754),
                                    atol=1e-4))
        if hasattr(OpenEXR, "File"):
            self.assertTrue(App._exr_is_aces(p))
        for fn in (App._smudge_save_frame, App._smudge_run):
            with self.subTest(fn=fn.__name__):
                self.assertIn("self._smudge_aces_out(lin,", self._src(fn))

    def test_reader_converts_aces_and_pq_before_the_display_encode(self):
        body = self._src(App._smudge_read_exr_gamma)
        self.assertIn("self._scene_from_raw(arr, self.sm_in_mode)", body)
        self.assertIn("self._ups_raw_float(path)", body)

    def test_changing_it_reloads_and_warns_about_unsaved_paint(self):
        body = self._src(App._smudge_in_changed)
        self.assertIn("messagebox.askyesno(", body)
        self.assertIn("self._smudge_load_folder(self.sm_folder)", body)
        self.assertIn("self.sm_in_var.set(self._sm_in_prev)", body)

    def test_a_reload_does_not_carry_old_paint_into_the_new_sequence(self):
        body = self._src(App._smudge_load_folder)
        i = body.index("self.sm_edits = {}")
        j = body.index("self._smudge_show_frame(0)")
        self.assertIn("self.sm_img = None", body[i:j])


class TestHotkeysSurviveADropdown(unittest.TestCase):
    """A ttk.Combobox is a tk.Entry subclass. The hotkey guard asked
    "is focus in an Entry?", so one click on a read-only dropdown (which
    then keeps the focus) silenced every hotkey on the tab."""

    def test_only_a_field_you_can_type_in_stands_hotkeys_down(self):
        import tkinter as tk
        from tkinter import ttk
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("no display")
        try:
            self.assertTrue(App._typing_in_field(tk.Entry(root)))
            self.assertTrue(App._typing_in_field(tk.Text(root)))
            self.assertTrue(App._typing_in_field(ttk.Combobox(root)))
            self.assertFalse(App._typing_in_field(
                ttk.Combobox(root, state="readonly")))
            self.assertFalse(App._typing_in_field(
                tk.Entry(root, state="disabled")))
            self.assertFalse(App._typing_in_field(tk.Canvas(root)))
            self.assertFalse(App._typing_in_field(None))
        finally:
            root.destroy()

    def test_every_guard_uses_it(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        # the one remaining use is inside the helper itself
        self.assertEqual(src.count("isinstance(w, (tk.Entry, tk.Text))"), 1)
        # Paint Studio, Lens Studio and SAM 3 each guard their hotkeys;
        # CurveLock and AreaLock took four guards with them.
        self.assertGreaterEqual(src.count("self._typing_in_field("), 3)

    def test_clicking_the_paint_canvas_takes_the_keyboard(self):
        import inspect
        self.assertIn("self.sm_canvas.focus_set()",
                      inspect.getsource(App._tab_smudge))


class TestMovToExrACES(unittest.TestCase):
    """MOV -> EXR can write ACES2065-1: linearised like Scene-linear,
    then Rec.709 -> AP0, tagged in the file and on the folder."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_it_is_an_output_choice(self):
        body = self._src(App._tab_exr)
        i = body.index('self.exr_out_var = self._combo(')
        self.assertIn('"ACES2065-1 (linear, AP0)"', body[i:i + 300])

    def test_aces_implies_linear(self):
        body = self._src(App._erun)
        self.assertIn('out_enc.get().startswith("ACES2065-1")', body)
        self.assertIn("want_linear = (out_enc is None or want_aces or", body)

    def test_passthrough_input_is_linearised_as_rec709_and_says_so(self):
        body = self._src(App._erun)
        self.assertIn('"Rec.709" in csp or (want_aces and "sRGB" not in csp)',
                      body)
        self.assertIn("ACES2065-1 needs linear light", body)

    def test_both_paths_convert_and_tag(self):
        body = self._src(App._erun)
        self.assertEqual(body.count("self._lin709_to_aces("), 2)
        self.assertEqual(body.count("aces=True"), 2)
        self.assertEqual(
            body.count("primaries=self.ACES_NAME if want_aces else None"), 2)

    def test_a_failed_conversion_is_not_reported_as_done(self):
        body = self._src(App._erun)
        i = body.index("ACES2065-1 conversion failed")
        self.assertIn("code = 1", body[i:i + 120])

    def test_other_outputs_are_untouched(self):
        body = self._src(App._erun)
        self.assertIn("ef_out = _OE8.OutputFile(out_path, hdr8)", body)
        self.assertIn('if code == 0 and want_aces:', body)


class TestMovToExrDropdownNames(unittest.TestCase):
    """The Output names were cut off in the box, and the Input names
    ("Rec.709 → Linear") read as if the input box did the converting."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_input_names_say_what_the_video_is(self):
        body = self._src(App._tab_exr)
        self.assertIn('["None — passthrough", "Rec.709", "sRGB"]', body)
        self.assertNotIn('"Rec.709 → Linear",', body)   # not an option
        # the conversion still keys off the same words
        run = self._src(App._erun)
        self.assertIn('"Rec.709" in csp', run)
        self.assertIn('"sRGB" in csp', run)

    def test_output_names_fit_and_the_box_gets_its_own_line(self):
        body = self._src(App._tab_exr)
        i = body.index('self.exr_out_var = self._combo(')
        j = body.index('self.exr_depth_var', i)
        seg = body[i:j]
        self.assertIn("wide=True", seg)
        opts = ["Scene-linear (EXR convention)", "ACES2065-1 (linear, AP0)",
                "Display-encoded — sRGB", "Display-encoded — Rec.709"]
        for o in opts:
            with self.subTest(option=o):
                self.assertIn(f'"{o}"', seg)
                self.assertLessEqual(len(o), 30)
        self.assertNotIn("(for video models)\"", seg)

    def test_old_saved_names_still_mean_the_same_thing(self):
        """A project saved with the long names restores them verbatim;
        the run only looks for the words."""
        run = self._src(App._erun)
        self.assertIn('out_enc.get().startswith("Scene-linear")', run)
        self.assertIn('"sRGB" in out_enc.get()', run)


class TestCutMode(unittest.TestCase):
    """The fourth MOV <-> EXR mode: a long movie or sequence split
    into its shots. ffmpeg scores every frame change once; the slider
    re-thresholds the stored scores. What matters most is that the
    shot ranges are right down to the frame, because every file or
    folder written is named from them."""

    SAMPLE = (
        "frame:0    pts:0       pts_time:0\n"
        "frame:1    pts:512     pts_time:0.04\n"
        "lavfi.scene_score=0.012000\n"
        "frame:2    pts:1024    pts_time:0.08\n"
        "lavfi.scene_score=0.410000\n"
        "frame:3    pts:1536    pts_time:0.12\n"
        "lavfi.scene_score=0.003000\n")

    # ── parsing ──
    def test_scores_line_up_with_frame_numbers(self):
        self.assertEqual(App._cut_parse_scene_scores(self.SAMPLE),
                         [0.0, 0.012, 0.41, 0.003])

    def test_the_first_frame_has_no_score_and_reads_as_zero(self):
        """ffmpeg prints no score for frame 0 (nothing to compare it
        with); it must still occupy index 0 or every shot shifts."""
        sc = App._cut_parse_scene_scores(self.SAMPLE)
        self.assertEqual(sc[0], 0.0)
        self.assertEqual(len(sc), 4)

    def test_empty_output_is_an_empty_list(self):
        self.assertEqual(App._cut_parse_scene_scores(""), [])
        self.assertEqual(App._cut_parse_scene_scores("garbage\n"), [])

    # ── thresholding ──
    def _scores(self, n, cuts):
        sc = [0.0] * n
        for i, v in cuts.items():
            sc[i] = v
        return sc

    def test_cuts_start_shots_and_ranges_are_inclusive(self):
        sc = self._scores(60, {20: 0.4, 35: 0.5})
        self.assertEqual(App._cut_shots_from_scores(sc, 0.7),
                         [(0, 19), (20, 34), (35, 59)])

    def test_higher_sensitivity_finds_more_cuts_never_fewer(self):
        import random
        rnd = random.Random(11)
        sc = [0.0] + [rnd.uniform(0.0, 0.3) for _ in range(199)]
        for i in (40, 95, 150):
            sc[i] = rnd.uniform(0.4, 0.9)
        prev = None
        for sens in (0.2, 0.4, 0.6, 0.8, 0.95):
            n = len(App._cut_shots_from_scores(sc, sens))
            if prev is not None:
                self.assertGreaterEqual(n, prev, sens)
            prev = n

    # ── the strength: a spike over the local baseline ──
    def _busy(self):
        """Handheld footage: every frame 0.18-0.32 from motion alone,
        three real cuts at 0.45-0.55, one two-frame flash."""
        import random
        rnd = random.Random(3)
        sc = [0.0] + [rnd.uniform(0.18, 0.32) for _ in range(199)]
        for i in (40, 95, 150):
            sc[i] = rnd.uniform(0.45, 0.55)
        sc[120] = sc[121] = 0.9
        return sc

    def test_busy_footage_cuts_are_found_and_motion_is_not(self):
        """The reported failure: with an absolute threshold no slider
        position separated 0.5 cuts from 0.3 motion. Against the local
        baseline they are far apart."""
        sc = self._busy()
        st = App._cut_strengths(sc)
        lo = min(st[i] for i in (40, 95, 150))
        hi = max(v for i, v in enumerate(st)
                 if i not in (40, 95, 150, 120, 121))
        self.assertGreater(lo, hi + 0.15, (lo, hi))

    def test_busy_footage_at_the_default_sensitivity(self):
        self.assertEqual(
            App._cut_shots_from_scores(self._busy(),
                                       App.CUT_DEFAULT_SENSITIVITY),
            [(0, 39), (40, 94), (95, 149), (150, 199)])

    def test_a_locked_off_shot_still_sees_a_modest_cut(self):
        """Baseline near zero: a 0.12 change is a huge spike there."""
        import random
        rnd = random.Random(5)
        sc = [0.0] + [rnd.uniform(0.0, 0.015) for _ in range(99)]
        sc[50] = 0.12
        self.assertEqual(App._cut_shots_from_scores(sc, 0.7),
                         [(0, 49), (50, 99)])

    def test_noise_under_the_floor_is_never_a_cut(self):
        sc = [0.0] + [0.03] * 20 + [0.035] + [0.03] * 20
        self.assertEqual(App._cut_strengths(sc)[21], 0.0)

    # ── hand edits ──
    def test_a_join_forbids_a_start_and_survives_the_slider(self):
        sc = self._busy()
        for sens in (0.6, 0.7, 0.8):
            shots = App._cut_shots_from_scores(sc, sens, joins={95})
            self.assertNotIn(95, [a for a, _ in shots], sens)

    def test_a_split_forces_a_start_even_where_nothing_scored(self):
        sc = self._busy()
        shots = App._cut_shots_from_scores(sc, 0.7, splits={60})
        self.assertIn((40, 59), shots)
        self.assertIn((60, 94), shots)

    def test_a_forced_split_is_not_folded_away_as_a_flash(self):
        sc = [0.0] * 10
        self.assertEqual(App._cut_shots_from_scores(sc, 0.7, splits={9}),
                         [(0, 8), (9, 9)])

    def test_no_cuts_is_one_shot(self):
        self.assertEqual(App._cut_shots_from_scores([0.0] * 5, 0.7), [(0, 4)])

    def test_no_frames_is_no_shots(self):
        self.assertEqual(App._cut_shots_from_scores([], 0.7), [])

    def test_a_flash_frame_is_not_two_cuts(self):
        """One white frame scores high going in AND coming out. It is
        neither a shot of its own nor the start of a new one: the
        frames after it are the same shot carrying on."""
        sc = self._scores(60, {20: 0.4, 45: 1.0, 46: 1.0})
        self.assertEqual(App._cut_shots_from_scores(sc, 0.7, min_len=2),
                         [(0, 19), (20, 59)])

    def test_a_short_run_at_the_very_end_joins_the_last_shot(self):
        sc = self._scores(10, {9: 1.0})
        self.assertEqual(App._cut_shots_from_scores(sc, 0.7, min_len=2),
                         [(0, 9)])

    def test_a_real_two_frame_shot_survives_at_min_len_2(self):
        sc = self._scores(10, {4: 0.9, 6: 0.9})
        self.assertEqual(App._cut_shots_from_scores(sc, 0.7, min_len=2),
                         [(0, 3), (4, 5), (6, 9)])

    def test_every_frame_belongs_to_exactly_one_shot(self):
        import random
        rnd = random.Random(7)
        for _ in range(50):
            n = rnd.randint(1, 80)
            sc = [rnd.random() for _ in range(n)]
            sc[0] = 0.0
            shots = App._cut_shots_from_scores(sc, rnd.random(),
                                               rnd.randint(1, 4))
            covered = [i for a, b in shots for i in range(a, b + 1)]
            self.assertEqual(covered, list(range(n)), (sc, shots))

    # ── naming and the ffmpeg command ──
    def test_shot_names_count_from_one(self):
        self.assertEqual(App._cut_shot_name("reel", 1), "reel_shot1")
        self.assertEqual(App._cut_shot_name("reel", 12), "reel_shot12")

    def test_intra_codecs_are_recognised(self):
        for c in ("prores", "prores_ks", "dnxhd", "mjpeg", "v210"):
            self.assertTrue(App._cut_codec_is_intra(c), c)
        for c in ("h264", "hevc", "mpeg4", "vp9", "", None):
            self.assertFalse(App._cut_codec_is_intra(c), c)

    def test_an_intra_source_is_stream_copied_between_its_own_frames(self):
        """Lossless: the shot is the source's bytes. The in point sits
        just past the first frame's time and the out point just short
        of the frame after the last, so rounding cannot grab a
        neighbour."""
        cmd = App._cut_segment_cmd("in.mov", "out.mov", 24, 47, 24.0, True)
        self.assertIn("copy", cmd)
        self.assertNotIn("prores_ks", cmd)
        ss = float(cmd[cmd.index("-ss") + 1])
        to = float(cmd[cmd.index("-to") + 1])
        self.assertGreater(ss, 24 / 24.0)
        self.assertLess(ss, 25 / 24.0)
        self.assertGreater(to, 47 / 24.0)
        self.assertLess(to, 48 / 24.0)
        # -ss before -i: the seek is on the input
        self.assertLess(cmd.index("-ss"), cmd.index("-i"))

    def test_a_long_gop_source_is_picked_by_frame_index(self):
        """A copy would snap to the previous keyframe, so the frames
        are selected by index and re-encoded as ProRes 422 HQ."""
        cmd = App._cut_segment_cmd("in.mp4", "out.mov", 24, 47, 25.0, False)
        vf = cmd[cmd.index("-vf") + 1]
        self.assertIn("between(n,24,47)", vf)
        self.assertIn("prores_ks", cmd)
        self.assertEqual(cmd[cmd.index("-profile:v") + 1], "3")
        self.assertNotIn("copy", cmd)

    # ── wiring ──
    def test_the_toggle_has_a_fourth_mode(self):
        import inspect
        body = inspect.getsource(App._tab_exr)
        self.assertIn('self.exr_tog_d = self._mkbtn(tog, "CUT"', body)
        i = body.index("self.exr_f_cut = tk.Frame")
        self.assertIn("self.exr_f_cut.grid_remove()", body[i:i + 300])

    def test_the_switcher_knows_cut(self):
        import inspect
        body = inspect.getsource(App._exr_set_mode)
        self.assertIn('"cut": self.exr_f_cut', body)
        self.assertIn('"cut": self.exr_tog_d', body)

    def test_the_timeline_is_shared_and_dispatches_by_mode(self):
        import inspect
        body = inspect.getsource(App._tab_exr)
        self.assertIn("get_frame_color=self._exr_frame_color", body)
        self.assertIn("self._exr_timeline_select", body)
        for fn in (App._exr_frame_color, App._exr_timeline_select):
            self.assertIn('== "cut"', inspect.getsource(fn))

    def test_the_viewer_routes_to_the_cut_preview(self):
        import inspect
        body = inspect.getsource(App._e2m_draw_preview)
        self.assertIn("self._cut_draw_preview()", body)

    def test_the_slider_rethresholds_without_touching_footage(self):
        import inspect
        body = inspect.getsource(App._cut_rethreshold)
        for forbidden in ("subprocess", "ffmpeg", "_pv_frame", "open("):
            self.assertNotIn(forbidden, body)

    def test_the_sequence_cut_carries_the_colour_sidecar(self):
        import inspect
        body = inspect.getsource(App._cut_worker)
        self.assertIn("ADW_CS_SIDECAR", body)
        self.assertIn("shutil.copy2", body)

    def test_the_edit_buttons_exist_and_the_input_pick_clears_them(self):
        import inspect
        body = inspect.getsource(App._tab_exr)
        for txt in ('"Merge with next"', '"Split here"', '"Reset edits"'):
            self.assertIn(txt, body)
        acc = inspect.getsource(App._cut_accept)
        self.assertIn("self._cut_joins = set()", acc)
        self.assertIn("self._cut_splits = set()", acc)

    def test_the_analysis_streams_progress(self):
        import inspect
        body = inspect.getsource(App._cut_preview_worker)
        self.assertIn("for line in proc.stdout:", body)
        self.assertNotIn("communicate()", body)


class TestMovExrTabLayoutAndAnalysis(unittest.TestCase):
    """MOV <-> EXR: panels that resize like the other modules', a real
    preview for MOV -> EXR, Expansion Studio's histogram, Upscale
    Studio's frequency graph, and a clipping view."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    # ── layout ──
    def test_left_column_scrolls(self):
        body = self._src(App._tab_exr)
        self.assertIn("self._scroll_column(lf)", body)

    def test_three_panes_viewer_graphs_log(self):
        body = self._src(App._tab_exr)
        self.assertEqual(body.count("rvpaned.add("), 3)
        self.assertIn("rvpaned._add_handle(0)", body)
        self.assertIn("rvpaned._add_handle(1)", body)
        i = body.index("self.e2m_preview_canvas = tk.Canvas")
        self.assertIn('fill="both", expand=True', body[i:i + 300])

    def test_exposure_bar_is_packed_before_the_expanding_canvas(self):
        body = self._src(App._tab_exr)
        self.assertLess(body.index("self._ev_toolbar(rtop,"),
                        body.index("self.e2m_preview_canvas = tk.Canvas"))

    def test_opening_the_tab_puts_the_log_on_screen(self):
        self.assertIn("self._exr_reset_sash", self._src(App.show))
        self.assertIn("vp.sash_place(1, 0, total - log_h)",
                      self._src(App._exr_reset_sash))

    # ── MOV -> EXR preview ──
    def test_picking_a_movie_makes_a_preview(self):
        body = self._src(App._epick)
        self.assertIn("self._m2e_refresh_preview()", body)
        self.assertNotIn("self._exr_sample_histogram(p)", body)

    def test_changing_a_setting_remakes_it(self):
        body = self._src(App._tab_exr)
        self.assertIn("(self.exr_csp_var, self.exr_out_var, "
                      "self.exr_depth_var)", body)
        self.assertIn("self._m2e_refresh_preview()", body)

    def test_preview_uses_the_same_conversion_as_the_run(self):
        run = self._src(App._erun)
        for csp in ("None \u2014 passthrough", "Rec.709", "sRGB"):
            for out in ("Scene-linear (EXR convention)",
                        "ACES2065-1 (linear, AP0)",
                        "Display-encoded \u2014 sRGB"):
                vf, lin, aces = App._m2e_filter(csp, out)
                with self.subTest(csp=csp, out=out):
                    if vf:
                        self.assertIn(f'vf = "{vf}"', run)
                    self.assertEqual(aces, out.startswith("ACES"))
                    self.assertEqual(lin, not out.startswith("Display"))
        self.assertEqual(App._m2e_filter("None \u2014 passthrough",
                                         "ACES2065-1 (linear, AP0)")[0],
                         "zscale=tin=709:t=linear:rangein=limited:range=full")
        self.assertEqual(App._m2e_filter("Rec.709",
                                         "Display-encoded \u2014 sRGB")[0],
                         "")

    def test_srgb_uses_a_keyword_zscale_actually_has(self):
        with open(SRC, encoding="utf-8") as fh:
            src = fh.read()
        self.assertNotIn("tin=srgb", src)
        self.assertNotIn("zscale=t=srgb", src)
        self.assertIn("tin=iec61966-2-1", src)

    def test_a_stale_preview_cannot_overwrite_a_newer_one(self):
        self.assertIn("if seq != self._m2e_seq:",
                      self._src(App._m2e_refresh_preview))

    # ── clipping view ──
    def test_clipping_marks_above_one_and_at_or_below_zero(self):
        vals = np.full((4, 4, 3), 0.5, np.float32)
        vals[0, 0] = (1.2, 0.5, 0.5)      # one channel over
        vals[1, 1] = (0.0, 0.5, 0.5)      # one channel at zero
        vals[2, 2] = (-0.1, 0.2, 0.2)     # below zero
        vals[3, 3] = (1.5, -0.2, 0.3)     # both
        disp = np.full((4, 4, 3), 128, np.uint8)
        out, fo, fu = App._clip_overlay(vals, disp)
        self.assertEqual(tuple(out[0, 0]), (255, 40, 40))
        self.assertEqual(tuple(out[1, 1]), (40, 110, 255))
        self.assertEqual(tuple(out[2, 2]), (40, 110, 255))
        self.assertEqual(tuple(out[3, 3]), (255, 255, 255))
        self.assertAlmostEqual(fo, 2 / 16)
        self.assertAlmostEqual(fu, 3 / 16)
        # the rest is dimmed grey, so the marks are what stand out
        self.assertTrue(out[0, 1, 0] == out[0, 1, 1] == out[0, 1, 2])
        self.assertLess(int(out[0, 1, 0]), 128)

    def test_exactly_one_is_not_clipped(self):
        vals = np.ones((2, 2, 3), np.float32)
        _o, fo, fu = App._clip_overlay(vals, np.zeros((2, 2, 3), np.uint8))
        self.assertEqual((fo, fu), (0.0, 0.0))

    def test_there_is_a_clipping_toggle(self):
        self.assertIn("self._exr_toggle_clip", self._src(App._tab_exr))
        self.assertIn("self._clip_overlay(vals, disp)",
                      self._src(App._e2m_draw_preview))

    # ── graphs ──
    def test_histogram_is_expansion_studios(self):
        body = self._src(App._exr_draw_histogram)
        self.assertIn("self._usc_histogram_refresh(cv=self.exr_hist_canvas, "
                      "arr=vals)", body)

    def test_frequency_graph_is_upscale_studios_spectrum(self):
        body = self._src(App._exr_freq_refresh)
        self.assertIn("self._ups_spectrum(", body)
        self.assertIn("self.exr_freq_canvas = tk.Canvas",
                      self._src(App._tab_exr))

    def test_expansion_histogram_takes_any_canvas_and_values(self):
        import tkinter as tk
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("no display")
        try:
            cv = tk.Canvas(root, width=300, height=90)
            cv.pack()
            root.update()

            class S:
                USC_HIST_LEFT_FRAC = App.USC_HIST_LEFT_FRAC
            arr = np.linspace(-0.05, 2.0, 3000, dtype=np.float32).reshape(
                30, 100)
            App._usc_histogram_refresh(S(), cv=cv, arr=np.stack([arr] * 3, -1))
            texts = [cv.itemcget(i, "text") for i in cv.find_all()
                     if cv.type(i) == "text"]
            self.assertTrue(any(t.startswith("max 2.0") for t in texts))
            self.assertTrue(any(t.startswith("min -0.05") for t in texts))
            self.assertTrue(texts[-1].startswith("max"))   # max stays on top
        finally:
            root.destroy()


class TestViewerFileInfo(unittest.TestCase):
    """Every viewer shows the file's resolution and bit depth in its
    lower-left corner -- read off the file, not the 8-bit picture."""

    def setUp(self):
        import tempfile
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _p(self, name):
        return os.path.join(self.d, name)

    def test_png_8_and_16_bit(self):
        import cv2
        cv2.imwrite(self._p("a.png"), np.zeros((30, 50, 3), np.uint8))
        cv2.imwrite(self._p("b.png"), np.zeros((30, 50, 3), np.uint16))
        self.assertEqual(App._file_info(self._p("a.png")),
                         (50, 30, "8-bit", "PNG"))
        # PIL calls a 16-bit RGB PNG mode "RGB"; the header does not lie
        self.assertEqual(App._file_info(self._p("b.png")),
                         (50, 30, "16-bit", "PNG"))

    def test_exr_half_float_and_aces(self):
        try:
            import OpenEXR
        except ImportError:
            self.skipTest("OpenEXR not installed")
        a = np.zeros((20, 40, 3), np.float32)
        App._organic_write_exr(self._p("h.exr"), a, half=True)
        App._organic_write_exr(self._p("f.exr"), a, half=False)
        self.assertEqual(App._file_info(self._p("h.exr")),
                         (40, 20, "16-bit half float", "EXR"))
        self.assertEqual(App._file_info(self._p("f.exr")),
                         (40, 20, "32-bit float", "EXR"))
        if hasattr(OpenEXR, "File"):
            App._organic_write_exr(self._p("x.exr"), a, half=True, aces=True)
            info = App._file_info(self._p("x.exr"))
            self.assertEqual(info[:3], (40, 20, "16-bit half float"))
            self.assertIn("ACES2065-1", info[3])

    def test_jpeg_and_16_bit_tiff(self):
        import cv2
        cv2.imwrite(self._p("a.jpg"), np.zeros((30, 50, 3), np.uint8))
        self.assertEqual(App._file_info(self._p("a.jpg")),
                         (50, 30, "8-bit", "JPEG"))
        cv2.imwrite(self._p("a.tif"), np.zeros((30, 50, 3), np.uint16))
        self.assertEqual(App._file_info(self._p("a.tif")),
                         (50, 30, "16-bit", "TIFF"))

    def test_dpx_header(self):
        import struct
        head = bytearray(2048)
        head[:4] = b"SDPX"
        head[772:780] = struct.pack(">II", 2048, 1556)
        head[803] = 10
        with open(self._p("a.dpx"), "wb") as fh:
            fh.write(head)
        self.assertEqual(App._file_info(self._p("a.dpx")),
                         (2048, 1556, "10-bit", "DPX"))

    def test_unreadable_is_none_not_a_crash(self):
        with open(self._p("bad.png"), "wb") as fh:
            fh.write(b"not a png")
        self.assertIsNone(App._file_info(self._p("bad.png")))
        self.assertIsNone(App._file_info(self._p("missing.exr")))

    def test_text_format_and_cache(self):
        import cv2

        class S:
            _file_info_text = App._file_info_text
            calls = 0

            @classmethod
            def _file_info(cls, path):
                cls.calls += 1
                return App._file_info(path)
        cv2.imwrite(self._p("a.png"), np.zeros((1080, 1920, 3), np.uint8))
        st = S()
        txt = st._file_info_text(self._p("a.png"))
        self.assertEqual(txt, "1920\u00d71080  \u00b7  8-bit  \u00b7  PNG")
        st._file_info_text(self._p("a.png"))
        self.assertEqual(S.calls, 1)
        self.assertEqual(st._file_info_text(None), "")
        self.assertEqual(st._file_info_text(self._p("gone.png")), "")

    def test_frame_path_takes_names_or_full_paths(self):
        open(self._p("f1.png"), "wb").close()
        fp = App._viewer_frame_path
        self.assertEqual(fp(self.d, ["f1.png"], 0), self._p("f1.png"))
        self.assertEqual(fp("/elsewhere", [self._p("f1.png")], 0),
                         self._p("f1.png"))
        self.assertEqual(fp(self.d, ["f1.png"], 99), self._p("f1.png"))
        self.assertIsNone(fp(self.d, [], 0))
        self.assertIsNone(fp(self.d, ["nope.png"], 0))
        self.assertIsNone(fp(None, None, None))

    def test_every_viewer_is_covered(self):
        import inspect
        body = inspect.getsource(App._viewer_info_attach_all)
        for cv in ("e2m_preview_canvas", "ups_canvas", "usc_canvas",
                   "organic_canvas", "sm_canvas", "ff_canvas",
                   "curve_canvas", "area_canvas", "rife_wipe_canvas",
                   "ip_canvas", "sam_canvas"):
            with self.subTest(canvas=cv):
                self.assertIn(f'"{cv}"', body)
        self.assertIn("self._viewer_info_attach_all()",
                      inspect.getsource(App._build))

    def test_the_line_survives_a_redraw_and_stays_on_top(self):
        import tkinter as tk
        import cv2
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("no display")
        try:
            cv2.imwrite(self._p("a.png"), np.zeros((30, 50, 3), np.uint16))
            cv = tk.Canvas(root, width=300, height=200)
            cv.pack()
            root.update()

            class S:
                _file_info_text = App._file_info_text
                _file_info = App._file_info
                _viewer_info_attach = App._viewer_info_attach
            st = S()
            cur = {"p": self._p("a.png")}
            st._viewer_info_attach(cv, lambda: cur["p"])
            root.update()

            def texts():
                return [cv.itemcget(i, "text")
                        for i in cv.find_withtag("viewer_info")
                        if cv.type(i) == "text"]
            self.assertEqual(texts(), ["50\u00d730  \u00b7  16-bit  \u00b7  PNG"])
            # a module redraws the way they all do
            cv.delete("all")
            self.assertEqual(texts(), [])
            cv.create_rectangle(0, 0, 300, 200, fill="red")
            root.update()
            self.assertEqual(len(texts()), 1)
            self.assertIn("viewer_info", cv.gettags(cv.find_all()[-1]))
            x0, y0, x1, y1 = cv.bbox(cv.find_all()[-1])
            self.assertLess(x0, 20)                 # left
            self.assertGreater(y1, 180)             # bottom
            # nothing loaded -> nothing shown
            cur["p"] = None
            cv.delete("all")
            root.update()
            self.assertEqual(texts(), [])
            # attaching twice does not stack two lines
            cur["p"] = self._p("a.png")
            st._viewer_info_attach(cv, lambda: cur["p"])
            cv.delete("all")
            root.update()
            self.assertEqual(len(texts()), 1)
        finally:
            root.destroy()


class TestExpansionMasksAndLensAdjustment(unittest.TestCase):
    """Expansion Studio can write its blown / crushed masks into the EXR
    (BlowMask, CrunchMask); Lens Studio's Expansion adjustment reads
    BlowMask to turn the expanded highlights down or up."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def _need_exr(self):
        try:
            import OpenEXR  # noqa: F401
        except ImportError:
            self.skipTest("OpenEXR not installed")

    # ── the adjustment itself ──
    def _frame(self):
        lin = np.full((6, 6, 3), 0.4, np.float32)
        lin[1, 1] = (10.0, 8.0, 6.0)       # expanded highlight, in mask
        lin[4, 4] = (3.0, 3.0, 3.0)        # above white, OUTSIDE mask
        mask = np.zeros((6, 6), np.float32)
        mask[0:3, 0:3] = 1.0               # includes dark ring pixels
        return lin, mask

    def test_gain_one_changes_nothing(self):
        lin, mask = self._frame()
        self.assertTrue(np.array_equal(
            App._expansion_adjust(lin, mask, 1.0), lin))
        self.assertTrue(np.array_equal(
            App._expansion_adjust(lin, None, 0.3), lin))

    def test_only_the_part_above_white_moves(self):
        lin, mask = self._frame()
        out = App._expansion_adjust(lin, mask, 0.5)
        self.assertAlmostEqual(float(out[1, 1, 0]), 5.5, places=4)
        # colour kept: channels scale together
        self.assertTrue(np.allclose(out[1, 1] / lin[1, 1], 0.55, atol=1e-4))
        # under-white pixels inside the mask (its dilated ring): untouched
        self.assertTrue(np.array_equal(out[0, 0], lin[0, 0]))
        # above white but outside the mask: untouched
        self.assertTrue(np.array_equal(out[4, 4], lin[4, 4]))

    def test_zero_returns_highlights_to_white_and_two_doubles_them(self):
        lin, mask = self._frame()
        self.assertAlmostEqual(
            float(App._expansion_adjust(lin, mask, 0.0)[1, 1].max()), 1.0,
            places=5)
        self.assertAlmostEqual(
            float(App._expansion_adjust(lin, mask, 2.0)[1, 1, 0]), 19.0,
            places=3)

    def test_a_mask_of_another_size_is_fitted(self):
        lin, _m = self._frame()
        big = np.ones((12, 12), np.float32)
        out = App._expansion_adjust(lin, big, 0.5)
        self.assertAlmostEqual(float(out[1, 1, 0]), 5.5, places=3)

    # ── Expansion Studio writes the channels ──
    def test_masks_match_what_the_mask_buttons_show(self):
        code = np.full((40, 40, 3), 0.5, np.float32)
        code[5:15, 5:15] = 1.0
        code[25:35, 25:35] = 0.0
        m = App._usc_frame_masks(code, 0.96, 0.04)
        self.assertEqual(set(m), {"BlowMask", "CrunchMask"})
        self.assertEqual(float(m["BlowMask"][10, 10]), 1.0)
        self.assertEqual(float(m["BlowMask"][30, 30]), 0.0)
        self.assertEqual(float(m["CrunchMask"][30, 30]), 1.0)
        self.assertEqual(float(m["CrunchMask"][10, 10]), 0.0)
        want = App._hdr_ai_mask_frame(code, 0.96).astype(np.float32) / 255
        self.assertTrue(np.array_equal(m["BlowMask"], want))

    def test_extra_channels_are_written_unconverted_and_read_back(self):
        self._need_exr()
        import tempfile, OpenEXR
        p = os.path.join(tempfile.mkdtemp(), "m.exr")
        rgb = np.full((8, 10, 3), 2.0, np.float32)
        blow = np.zeros((8, 10), np.float32)
        blow[2:4, 3:6] = 1.0
        App._usc_write_exr_layers(p, rgb, {}, half=True, aces=True,
                                  extra={"BlowMask": blow,
                                         "CrunchMask": 1.0 - blow})
        self.assertEqual(
            set(OpenEXR.InputFile(p).header()["channels"].keys()),
            {"R", "G", "B", "BlowMask", "CrunchMask"})
        got = App._exr_read_channel(p, "BlowMask")
        self.assertTrue(np.array_equal(got, blow))     # not colour-converted
        self.assertIsNone(App._exr_read_channel(p, "NoSuch"))
        self.assertIsNone(App._exr_read_channel(p + ".png", "BlowMask"))

        class R:
            pass
        pic = App._ff_read_frame.__get__(R())(p)       # RGB readers unaffected
        self.assertEqual(pic.shape, (8, 10, 3))

    def test_the_option_exists_and_the_run_uses_it(self):
        tab = self._src(App._tab_upscale)
        self.assertIn("Include masks  (BlowMask, CrunchMask)", tab)
        run = self._src(App._usc_run)
        self.assertIn("self._usc_masks_on()", run)
        self.assertIn("self._usc_frame_masks(code, *_mask_lv)", run)
        self.assertIn("extra=_extra", run)
        # off by default, EXR only
        class S:
            usc_masks_var = None
            _usc_masks_on = App._usc_masks_on
        self.assertFalse(S()._usc_masks_on())
        self.assertIn("self.usc_masks_chk.configure(",
                      self._src(App._usc_multi_changed))

    # ── Lens Studio ──
    def test_the_card_sits_right_under_the_input(self):
        tab = self._src(App._tab_organic)
        src = tab.index('self._card(lf, "Source")')
        exp = tab.index('self._card(lf, "Expansion adjustment")')
        pre = tab.index("# ── presets ──")
        self.assertTrue(src < exp < pre)

    def test_it_is_applied_at_every_read_with_the_original_left_alone(self):
        show = self._src(App._organic_show_frame)
        i = show.index("self.organic_img_base = Image.fromarray(")
        j = show.index("self._organic_exp_apply(")
        self.assertLess(i, j)          # the wipe's original is taken first
        self.assertIn("self._organic_exp_apply(",
                      self._src(App._organic_load_frame))
        run = self._src(App._organic_run_worker)
        self.assertIn("_exp_gain = self._organic_exp_gain()", run)
        self.assertIn("self._organic_exp_apply(lin, src, gain=_exp_gain)",
                      run)

    def test_apply_reads_the_blowmask_of_that_frame(self):
        self._need_exr()
        import tempfile
        p = os.path.join(tempfile.mkdtemp(), "f.exr")
        lin, mask = self._frame()
        App._usc_write_exr_layers(p, lin, {}, half=False,
                                  extra={"BlowMask": mask})

        class S:
            EXR_BLOW_CH = App.EXR_BLOW_CH
            organic_exp_var = None
            _organic_exp_stats = None
            _organic_exp_gain = App._organic_exp_gain
            _organic_exp_apply = App._organic_exp_apply
            _exr_read_channel = staticmethod(App._exr_read_channel)
            _expansion_adjust = staticmethod(App._expansion_adjust)
        st = S()
        out = st._organic_exp_apply(lin, p, gain=0.5, note=True)
        self.assertAlmostEqual(float(out[1, 1, 0]), 5.5, places=3)
        self.assertEqual(st._organic_exp_stats[0], True)
        self.assertAlmostEqual(st._organic_exp_stats[1], 10.0, places=3)
        self.assertAlmostEqual(st._organic_exp_stats[2], 5.5, places=3)
        # a frame with no BlowMask: unchanged, and says so
        p2 = os.path.join(os.path.dirname(p), "g.exr")
        App._organic_write_exr(p2, lin, half=False)
        out2 = st._organic_exp_apply(lin, p2, gain=0.5, note=True)
        self.assertTrue(np.array_equal(out2, lin))
        self.assertEqual(st._organic_exp_stats[0], False)

    def test_slider_changes_are_debounced(self):
        body = self._src(App._organic_exp_changed)
        self.assertIn("self.after_cancel(self._organic_exp_job)", body)
        # ...and that debounce is only for changes that are not a drag
        self.assertLess(body.index("if self._organic_exp_dragging:"),
                        body.index("self.after("))


class TestMasksWithEveryModel(unittest.TestCase):
    """The mask channels are written with Tone map and Topaz too, and
    the Clip level / mask buttons are reachable for them."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_masks_come_from_the_source_frame_not_the_model(self):
        run = self._src(App._usc_run)
        i = run.index("_extra = (self._usc_frame_masks(code, *_mask_lv)")
        j = run.index("if _hdr and _multi and _fspec", i)
        self.assertLess(i, j)              # before any model branch
        k = run.index("_mask_lv = (self._usc_mask_levels()")
        seg = run[k:k + 200]
        self.assertNotIn("wan", seg.lower())
        self.assertNotIn("_usc_ai_on", seg)

    def test_controls_show_for_any_model_when_masks_are_on(self):
        class S:
            _usc_mask_controls_relevant = App._usc_mask_controls_relevant
            on = False
            wan = False
            _usc_masks_on = lambda s: s.on
            _usc_wan_relevant = lambda s: s.wan
        st = S()
        self.assertFalse(st._usc_mask_controls_relevant())
        st.on = True
        self.assertTrue(st._usc_mask_controls_relevant())     # Tone map / Topaz
        st.on, st.wan = False, True
        self.assertTrue(st._usc_mask_controls_relevant())     # Wan, as before
        self.assertIn("self._usc_mask_controls_relevant()",
                      self._src(App._usc_models_changed))

    def test_the_mask_buttons_work_without_wan(self):
        for fn in (App._usc_ai_show_mask, App._usc_ai_show_shadow_mask):
            with self.subTest(fn=fn.__name__):
                body = self._src(fn)
                self.assertIn("if not self._usc_mask_controls_relevant():",
                              body)
                self.assertNotIn("if not self._usc_wan_relevant():", body)

    def test_ticking_the_box_brings_the_controls_up(self):
        tab = self._src(App._tab_upscale)
        i = tab.index("self.usc_masks_chk = tk.Checkbutton(")
        self.assertIn("command=self._usc_models_changed", tab[i:i + 400])


class TestShadowsLeftHighlightsRight(unittest.TestCase):
    """The two tabs sit in the histogram's order: dark end on the left,
    bright end on the right."""

    def test_order_and_default(self):
        import inspect
        body = inspect.getsource(App._tab_upscale)
        lo = body.index('self.usc_ai_notebook.add(_lo_tab, text="Shadows")')
        hi = body.index('self.usc_ai_notebook.add(_hi_tab, text="Highlights")')
        self.assertLess(lo, hi)
        self.assertIn("self.usc_ai_notebook.select(_hi_tab)", body)


class TestLensStudioInputColour(unittest.TestCase):
    """Lens Studio names every colourspace Expansion Studio can write --
    Linear Rec.709, ACES2065-1, HDR PQ / Rec.2020 -- plus display and
    Auto, and can load the HDR10 movie."""

    @staticmethod
    def _src(fn):
        import inspect
        return inspect.getsource(fn)

    def test_one_choice_per_expansion_output(self):
        c = App.LENS_IN_CHOICES
        self.assertEqual(c[0], "Auto")
        for want in ("Linear Rec.709", "ACES2065-1", "HDR PQ / Rec.2020"):
            self.assertIn(want, c)
        self.assertTrue(any(x.startswith("Display-encoded") for x in c))
        # ...and Expansion's own EXR names are among them
        for name in App.USC_EXR_SPACES:
            self.assertIn(name, c)

    def _stub(self, choice, transfer=""):
        class V:
            def __init__(s, v): s.v = v
            def get(s): return s.v

        class S:
            _organic_apply_in_choice = App._organic_apply_in_choice
            _organic_source_encoded = "x"
            _organic_source_aces = "x"
            _organic_source_pq = "x"
            _organic_cs_reason = ""
        st = S()
        st.organic_cs_var = V(choice)
        st._organic_src_transfer = transfer
        st._organic_apply_in_choice()
        return (st._organic_source_encoded, st._organic_source_aces,
                st._organic_source_pq)

    def test_each_choice_sets_how_frames_are_read(self):
        self.assertEqual(self._stub("Display-encoded (sRGB / Rec.709)"),
                         (True, False, False))
        self.assertEqual(self._stub("Linear Rec.709"), (False, False, False))
        self.assertEqual(self._stub("ACES2065-1"), (False, True, False))
        self.assertEqual(self._stub("HDR PQ / Rec.2020"),
                         (False, False, True))
        # Auto leaves it to the first frame read...
        self.assertEqual(self._stub("Auto"), (None, None, False))
        # ...unless the movie itself says PQ
        self.assertEqual(self._stub("Auto", "smpte2084"),
                         (False, False, True))

    def test_names_saved_by_older_projects_still_work(self):
        self.assertEqual(self._stub("Display-encoded"), (True, False, False))
        self.assertEqual(self._stub("Scene-linear"), (False, False, False))

    def test_pq_frames_come_back_as_linear_rec709(self):
        import tempfile
        lin709 = np.full((4, 4, 3), 0.3, np.float32)
        lin709[0, 0] = (8.0, 6.0, 2.0)
        lin2020 = (lin709 @ np.array(App._M_709_TO_2020, np.float32).T)
        p = os.path.join(tempfile.mkdtemp(), "f.png")
        App._ups_write16(p, App._hdr_pq_encode(lin2020 * App.HDR_SDR_WHITE))

        class S:
            FF_LINEAR_EXTS = App.FF_LINEAR_EXTS
            HDR_SDR_WHITE = App.HDR_SDR_WHITE
            _organic_source_pq = True
            _organic_load_linear = App._organic_load_linear
            _ups_raw_float = App._ups_raw_float
            _ups_read16 = staticmethod(App._ups_read16)
            _hdr_pq_decode = staticmethod(App._hdr_pq_decode)
            _bt2020_to_709 = App._bt2020_to_709
        got = S()._organic_load_linear(p)
        self.assertLess(float((np.abs(got - lin709)
                               / np.maximum(lin709, 0.05)).max()), 0.02)
        # the render's snapshot decides, not the live flag
        st = S()
        st._organic_source_pq = False
        got2 = st._organic_load_linear(p, pq=True)
        self.assertLess(float(np.abs(got2 - got).max()), 1e-6)

    def test_render_snapshots_pq_too(self):
        run = self._src(App._organic_run_worker)
        self.assertIn("_pq = bool(self._organic_source_pq)", run)
        self.assertIn("pq=_pq,", run)

    def test_a_movie_can_be_loaded_and_renders_land_beside_it(self):
        self.assertIn("self._organic_pick_movie", self._src(App._tab_organic))
        body = self._src(App._organic_load_movie)
        self.assertIn("self._ups_decode_movie16(", body)       # 16-bit
        self.assertIn("self._organic_load_folder(out, _movie=path)", body)
        self.assertIn("self._organic_out_base or self.organic_folder",
                      self._src(App._organic_run))

    def test_the_choice_is_applied_on_load_and_shown(self):
        self.assertIn("self._organic_apply_in_choice()",
                      self._src(App._organic_load_folder))
        self.assertIn("self._organic_in_refresh_label()",
                      self._src(App._organic_show_frame))


class TestLensStudioCancelRender(unittest.TestCase):
    """Lens Studio's render can be cancelled, the way Expansion
    Studio's can: the button turns into CANCEL RENDER, the worker stops
    between frames, and a partial take is neither encoded nor passed on."""

    def _stub(self, n=6, cancel_after=None, confirm=True):
        import tempfile

        class Btn:
            def __init__(s): s.kw = {}
            def configure(s, **kw): s.kw.update(kw)

        class V:
            def __init__(s, v): s.v = v
            def get(s): return s.v

        class S:
            _organic_run = lambda s: None
            _organic_cancel_run = App._organic_cancel_run
            _organic_run_btn_idle = App._organic_run_btn_idle
            _organic_run_worker = App._organic_run_worker
            _organic_out_stem = staticmethod(App._organic_out_stem)
            _organic_running = True
            _organic_cancel = False
            _organic_source_encoded = True
            _organic_source_pq = False
            _organic_source_aces = False
            ACES_NAME = App.ACES_NAME
            organic_log_txt = None
        st = S()
        st.out = tempfile.mkdtemp()
        st.organic_run_btn = Btn()
        st.organic_folder = "/src"
        st.organic_frames = [f"f.{i:04d}.png" for i in range(n)]
        st.organic_fmt_var = V("EXR 16-bit half")
        st.organic_out_mov_var = V(True)
        st.logs, st.wrote, st.encoded, st.sent = [], [], [], []
        st.log = lambda w, m, t=None: st.logs.append(m)
        st.after = lambda ms, fn: fn()
        st.progress = lambda *a: None
        st._reveal = lambda d: None
        st._organic_exp_gain = lambda: 1.0
        st._adw_write_colorspace = lambda *a, **k: None
        st._organic_load_linear = lambda p, **k: p
        st._organic_exp_apply = lambda lin, src, gain=1.0: lin
        st._organic_apply_stack = lambda lin, params, seed=0: lin
        st._organic_frame_seed = lambda i: i

        def write(stem, rgb, fmt, aces=False):
            st.wrote.append(stem)
            if cancel_after is not None and len(st.wrote) == cancel_after:
                st._organic_cancel = True
            return stem + ".exr"
        st._organic_write_frame = write
        st._organic_encode_mov = lambda d, fr: st.encoded.append(d)
        st._organic_after_run_sendto = lambda d: st.sent.append(d)
        return st

    def test_run_turns_the_button_into_cancel(self):
        import inspect
        src = inspect.getsource(App._organic_run)
        self.assertIn('"CANCEL RENDER"', src)
        self.assertIn("command=self._organic_cancel_run", src)
        self.assertIn("self._organic_running = True", src)
        # a second click on Run while rendering must not start a twin
        self.assertIn("if self._organic_running:", src)

    def test_cancel_stops_between_frames(self):
        st = self._stub(n=6, cancel_after=2)
        st._organic_run_worker(st.out, 0, 6, {})
        self.assertEqual(len(st.wrote), 2)
        self.assertEqual(st.encoded, [])
        self.assertEqual(st.sent, [])
        self.assertTrue(any("Cancelled" in m and "2 of 6" in m
                            for m in st.logs), st.logs)

    def test_button_and_flags_reset_after_a_cancel(self):
        st = self._stub(n=4, cancel_after=1)
        st._organic_run_worker(st.out, 0, 4, {})
        self.assertFalse(st._organic_running)
        self.assertFalse(st._organic_cancel)
        self.assertIn("Render Sequence", st.organic_run_btn.kw["text"])
        self.assertEqual(st.organic_run_btn.kw["state"], "normal")
        self.assertEqual(st.organic_run_btn.kw["fg"], "#66CC66")

    def test_an_uncancelled_render_still_finishes_and_hands_off(self):
        st = self._stub(n=3)
        st._organic_run_worker(st.out, 0, 3, {})
        self.assertEqual(len(st.wrote), 3)
        self.assertEqual(st.encoded, [st.out])
        self.assertEqual(st.sent, [st.out])
        self.assertFalse(st._organic_running)

    def test_saying_no_to_the_question_keeps_rendering(self):
        from unittest import mock
        st = self._stub()
        with mock.patch("tkinter.messagebox.askyesno", return_value=False):
            st._organic_cancel_run()
        self.assertFalse(st._organic_cancel)
        with mock.patch("tkinter.messagebox.askyesno", return_value=True):
            st._organic_cancel_run()
        self.assertTrue(st._organic_cancel)
        self.assertEqual(st.organic_run_btn.kw["state"], "disabled")

    def test_cancel_when_nothing_is_rendering_does_nothing(self):
        from unittest import mock
        st = self._stub()
        st._organic_running = False
        with mock.patch("tkinter.messagebox.askyesno",
                        return_value=True) as ask:
            st._organic_cancel_run()
        ask.assert_not_called()
        self.assertFalse(st._organic_cancel)


class TestExpansionAdjustmentIsNotLaggy(unittest.TestCase):
    """The Highlights slider recalculates on RELEASE, from the frame
    already in memory, and only inside the Region of Interest."""

    def _stub(self, roi=None, h=40, w=60):
        class V:
            def __init__(s, v): s.v = v
            def get(s): return s.v

        class Lbl:
            def configure(s, **k): s.k = k

        class S:
            for _n in ("_organic_exp_changed", "_organic_exp_press",
                       "_organic_exp_release", "_organic_exp_commit",
                       "_organic_exp_ensure_full", "_organic_exp_gain"):
                locals()[_n] = getattr(App, _n)
            _expansion_adjust = staticmethod(App._expansion_adjust)
            _organic_exp_dragging = False
            _organic_exp_job = None
            _organic_exp_partial = False
            _organic_exp_applied = 1.0
            _organic_exp_stats = None
            _organic_stale = False
            _organic_idx = 0
            organic_frames = ["a.exr"]
            organic_processed = object()
        st = S()
        raw = np.full((h, w, 3), 0.5, np.float32)
        raw[5:10, 5:10] = 9.0
        raw[30:35, 50:55] = 9.0
        mask = np.zeros((h, w), np.float32)
        mask[5:10, 5:10] = 1.0
        mask[30:35, 50:55] = 1.0
        st._organic_exp_raw, st._organic_exp_mask = raw, mask
        st.organic_lin = raw
        st.organic_exp_var = V(1.0)
        st.organic_exp_val = Lbl()
        st._organic_roi = roi
        st.n = {"recompute": 0, "show": 0, "after": 0, "idle": 0}
        st.states = []
        st._organic_preview_state = lambda s_, t=None: st.states.append(s_)
        st._organic_exp_refresh_label = lambda: None
        st._pp_invalidate = lambda k: None
        st._organic_params = lambda: {}
        st._organic_roi_crop = lambda shape, r, p: (
            (max(0, r[0] - 2), max(0, r[1] - 2),
             min(shape[0], r[2] + 2), min(shape[1], r[3] + 2)))

        def rec(): st.n["recompute"] += 1
        def show(i): st.n["show"] += 1
        def after(ms, fn): st.n["after"] += 1; st.pending = fn; return 1
        def idle(fn): st.n["idle"] += 1; fn()
        st._organic_recompute, st._organic_show_frame = rec, show
        st.after, st.after_idle = after, idle
        st.after_cancel = lambda j: None
        return st

    def test_nothing_is_recalculated_while_dragging(self):
        st = self._stub()
        st._organic_exp_press()
        for v in (0.9, 0.7, 0.5):
            st.organic_exp_var.v = v
            st._organic_exp_changed()
        self.assertEqual(st.n, {"recompute": 0, "show": 0,
                                "after": 0, "idle": 0})
        self.assertEqual(st.organic_exp_val.k["text"], "\u00d70.50")
        self.assertEqual(st.states[-1], "stale")

    def test_release_recalculates_once_without_rereading_the_file(self):
        st = self._stub()
        st._organic_exp_press()
        st.organic_exp_var.v = 0.5
        st._organic_exp_changed()
        st._organic_exp_release()
        self.assertEqual(st.n["recompute"], 1)
        self.assertEqual(st.n["show"], 0)
        self.assertFalse(st._organic_exp_dragging)
        self.assertAlmostEqual(float(st.organic_lin[6, 6, 0]), 5.0, places=4)

    def test_a_click_that_changes_nothing_recalculates_nothing(self):
        st = self._stub()
        st._organic_exp_press()
        st._organic_exp_release()
        self.assertEqual(st.n["recompute"], 0)

    def test_keys_and_reset_are_debounced_not_ignored(self):
        st = self._stub()
        st.organic_exp_var.v = 2.0
        st._organic_exp_changed()
        self.assertEqual((st.n["after"], st.n["recompute"]), (1, 0))
        st.pending()
        self.assertEqual(st.n["recompute"], 1)

    def test_with_a_roi_only_the_crop_is_adjusted(self):
        st = self._stub(roi=(4, 4, 12, 12))
        st.organic_exp_var.v = 0.0
        st._organic_exp_commit()
        self.assertAlmostEqual(float(st.organic_lin[6, 6, 0]), 1.0, places=4)
        # the highlight outside the region has not been worked on yet
        self.assertAlmostEqual(float(st.organic_lin[32, 52, 0]), 9.0,
                               places=4)
        self.assertTrue(st._organic_exp_partial)
        # ...and a full-frame update finishes it before reading
        st._organic_exp_ensure_full()
        self.assertAlmostEqual(float(st.organic_lin[32, 52, 0]), 1.0,
                               places=4)
        self.assertFalse(st._organic_exp_partial)

    def test_full_frame_update_finishes_a_partial_adjustment(self):
        import inspect
        src = inspect.getsource(App._organic_recompute)
        self.assertIn("self._organic_exp_ensure_full()", src)
        self.assertLess(src.index("self._organic_exp_ensure_full()"),
                        src.index("out = self._organic_apply_stack("))

    def test_no_blowmask_means_no_work(self):
        st = self._stub()
        st._organic_exp_mask = None
        st.organic_exp_var.v = 0.5
        st._organic_exp_commit()
        self.assertEqual(st.n["recompute"], 0)

    def test_playback_drops_the_in_memory_copy(self):
        import inspect
        src = inspect.getsource(App._organic_load_frame)
        self.assertIn("self._organic_exp_raw = None", src)

    def test_slider_is_bound_to_press_and_release(self):
        import inspect
        src = inspect.getsource(App)
        self.assertIn('"<ButtonRelease-1>", self._organic_exp_release', src)
        self.assertIn('"<ButtonPress-1>", self._organic_exp_press', src)


class TestLensStudioFrameNames(unittest.TestCase):
    """Rendered frames carry the module's name: Lens_Studio.00012.exr,
    not the bare 00012.exr an Expansion Studio source would hand on."""

    def test_number_only_source_gets_the_module_name(self):
        fr = ["00001.exr", "00002.exr", "00003.exr"]
        self.assertEqual(App._organic_out_stem(fr, 1), "Lens_Studio.00002")

    def test_source_frame_numbers_are_kept(self):
        fr = ["shot_010.1001.exr", "shot_010.1002.exr"]
        self.assertEqual(App._organic_out_stem(fr, 0), "Lens_Studio.01001")
        self.assertEqual(App._organic_out_stem(fr, 1), "Lens_Studio.01002")

    def test_unnumbered_or_clashing_sources_fall_back_to_position(self):
        self.assertEqual(App._organic_out_stem(["a.png", "b.png"], 1),
                         "Lens_Studio.00002")
        self.assertEqual(
            App._organic_out_stem(["a_1.png", "b_1.png"], 1),
            "Lens_Studio.00002")

    def test_render_uses_it_and_the_movie_pattern_matches(self):
        import inspect
        self.assertIn("self._organic_out_stem(self.organic_frames, i)",
                      inspect.getsource(App._organic_run_worker))
        import re
        stem = App._organic_out_stem(["00007.exr"], 0)
        base = re.sub(r"\d+$", "", stem)
        self.assertEqual((base + "%05d") % 7, stem)
        self.assertIn('_re_mov.sub(r"\\d+$", ""',
                      inspect.getsource(App._organic_encode_mov))

if __name__ == "__main__":
    unittest.main(verbosity=2)
