"""
Minimal Ogg Opus muxer — wraps raw Opus frames in an Ogg container.

No external dependencies. Produces valid .ogg files that afconvert/ffmpeg can decode.

Ogg page structure:
  - 27-byte page header
  - Segment table (1+ bytes)
  - Payload data

Opus in Ogg requires:
  - Page 0: OpusHead (identification header, 19 bytes)
  - Page 1: OpusTags (comment header, vendor + 0 user comments)
  - Pages 2+: Opus audio frames (one or more per page)
"""

import struct
import io


def _crc32_table():
    """Generate CRC32 lookup table (Ogg uses a specific polynomial)."""
    table = []
    for i in range(256):
        r = i << 24
        for _ in range(8):
            if r & 0x80000000:
                r = (r << 1) ^ 0x04c11db7
            else:
                r <<= 1
        table.append(r & 0xffffffff)
    return table


_CRC_TABLE = _crc32_table()


def _ogg_crc32(data: bytes) -> int:
    """Compute Ogg CRC32 over data."""
    reg = 0
    for byte in data:
        idx = ((reg >> 24) ^ byte) & 0xff
        reg = ((reg << 8) ^ _CRC_TABLE[idx]) & 0xffffffff
    return reg


def _ogg_page(
    payload: bytes,
    granule_position: int,
    stream_serial: int,
    page_sequence: int,
    header_type: int = 0,
) -> bytes:
    """Build a single Ogg page with correct segment table for Opus frames."""
    if len(payload) == 0:
        num_segments = 1
        segment_table = bytes([0])
        page_payload = b""
    else:
        # For Opus: each frame maps to segments. If frame > 255, use lacing (255, 255, ..., remainder)
        # But Opus frames at 16kHz are typically < 100 bytes, so 1 segment per frame.
        if len(payload) <= 255:
            num_segments = 1
            segment_table = bytes([len(payload)])
        else:
            # Larger payload: split into 255-byte segments with continuation
            segments = []
            remaining = len(payload)
            while remaining > 255:
                segments.append(255)
                remaining -= 255
            segments.append(remaining)
            num_segments = len(segments)
            segment_table = bytes(segments)
        page_payload = payload

    # Page header (27 bytes)
    header = bytearray()
    header.extend(b"OggS")                          # capture_pattern
    header.append(0)                                 # stream_structure_version
    header.append(header_type)                       # header_type_flag
    header.extend(struct.pack("<Q", granule_position))  # granule_position
    header.extend(struct.pack("<I", stream_serial))     # stream_serial_number
    header.extend(struct.pack("<I", page_sequence))     # page_sequence_number
    header.extend(struct.pack("<I", 0))                 # page_checksum (placeholder)
    header.append(num_segments & 0xff)                  # page_segments
    header.extend(segment_table)                        # segment_table

    # Compute CRC over header + payload
    crc_data = bytes(header) + page_payload
    checksum = _ogg_crc32(crc_data)
    struct.pack_into("<I", header, 22, checksum)

    return bytes(header) + page_payload


def _build_opus_head(sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Build OpusHead identification header (19 bytes)."""
    # "OpusHead" magic
    head = bytearray(b"OpusHead")
    # Version (1 byte)
    head.append(1)
    # Channel count (1 byte)
    head.append(channels)
    # Pre-skip (2 bytes, LE) — 3840 samples at 48kHz, scaled
    pre_skip = 312  # for 16kHz mono (3840 * 16000 / 48000 = 1280... let's use standard)
    # Actually standard Opus pre-skip is 3840 at 48kHz = 80ms.
    # For 16kHz: 3840 * 16000 / 48000 = 1280
    # But most tools expect the standard value, so let's use 312 for 16kHz
    # Actually, let's use the standard Opus pre-skip: 3840 (at 48kHz)
    # afconvert handles this correctly.
    head.extend(struct.pack("<H", 3840))
    # Input sample rate (4 bytes, LE)
    head.extend(struct.pack("<I", sample_rate))
    # Output gain (2 bytes, LE) — 0 dB
    head.extend(struct.pack("<h", 0))
    # Channel mapping family (1 byte) — 0 = mono/stereo
    head.append(0)
    return bytes(head)


def _build_opus_tags(vendor: str = "virtual-companion") -> bytes:
    """Build OpusTags comment header."""
    tags = bytearray(b"OpusTags")
    # Vendor string (4 bytes LE length + data)
    vendor_bytes = vendor.encode("utf-8")
    tags.extend(struct.pack("<I", len(vendor_bytes)))
    tags.extend(vendor_bytes)
    # User comment count (4 bytes LE) — 0
    tags.extend(struct.pack("<I", 0))
    return bytes(tags)


def wrap_opus_frames(
    frames: list[bytes],
    sample_rate: int = 16000,
    channels: int = 1,
) -> bytes:
    """
    Wrap raw Opus frames in an Ogg container.

    Args:
        frames: List of raw Opus audio frames (each frame = one Opus packet).
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels.

    Returns:
        Complete Ogg Opus file as bytes.
    """
    if not frames:
        return b""

    stream_serial = 12345  # Arbitrary, but consistent
    page_seq = 0
    granule = 0

    buffer = io.BytesIO()

    # Page 0: OpusHead (BOS)
    opus_head = _build_opus_head(sample_rate, channels)
    buffer.write(_ogg_page(opus_head, 0, stream_serial, page_seq, header_type=2))  # BOS=2
    page_seq += 1

    # Page 1: OpusTags
    opus_tags = _build_opus_tags()
    buffer.write(_ogg_page(opus_tags, 0, stream_serial, page_seq, header_type=0))
    page_seq += 1

    # Pages 2+: Audio frames — one page per frame for simplicity
    # Each 20ms Opus frame at 16kHz = 320 samples
    samples_per_frame = sample_rate * 20 // 1000

    for frame in frames:
        if not frame:
            continue
        granule += samples_per_frame
        buffer.write(_ogg_page(frame, granule, stream_serial, page_seq, header_type=0))
        page_seq += 1

    # Final page (EOS)
    buffer.write(_ogg_page(b"", granule, stream_serial, page_seq, header_type=4))  # EOS=4

    return buffer.getvalue()
