import spiceypy as spice
# written by ChatGPT
def frame_coverage_ranges(frame_name, out_timesys="TDB"):
    frcode = spice.namfrm(frame_name)
    if frcode == 0:
        raise ValueError(f"Unknown frame: {frame_name}")

    cent, frclss, clssid = spice.frinfo(frcode)

    ranges = []

    if frclss == 2:  # PCK frame
        for i in range(spice.ktotal("PCK")):
            file, ktype, source, handle = spice.kdata(i, "PCK")
            cover = spice.pckcov(file, clssid)
            for j in range(spice.wncard(cover)):
                beg, end = spice.wnfetd(cover, j)
                ranges.append((spice.et2utc(beg, "C", 3), spice.et2utc(end, "C", 3)))

    elif frclss == 3:  # CK frame
        for i in range(spice.ktotal("CK")):
            file, ktype, source, handle = spice.kdata(i, "CK")
            cover = spice.ckcov(file, clssid, False, "INTERVAL", 0.0, out_timesys)
            for j in range(spice.wncard(cover)):
                beg, end = spice.wnfetd(cover, j)
                if out_timesys.upper() == "TDB":
                    ranges.append((spice.et2utc(beg, "C", 3), spice.et2utc(end, "C", 3)))
                else:
                    ranges.append((beg, end))
    else:
        # TK / inertial / many dynamic frames don't have a file coverage window
        return []

    return ranges