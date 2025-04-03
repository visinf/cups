def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix) :]


def remove_suffix(text, suffix):
    if text.endswith(suffix):
        return text[: -len(suffix)]
    else:
        return text


def remove_suffixes(text, suffixes):
    for suffix in suffixes:
        text = remove_suffix(text, suffix)
    return text
