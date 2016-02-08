import sys
import itertools

spinner = itertools.cycle(["`", "*", ";", ","])


def title(text):
    """Prints title

    Parameters
    ----------
    text : str
        Title

    Returns
    -------
    Nothing

    """

    print "--------------------------------"
    print text
    print "--------------------------------"


def section_header(text):
    """Prints section header

    Parameters
    ----------
    text : str
        Section header

    Returns
    -------
    Nothing

    """

    print " "
    print text
    print "================================"


def foot():
    """Prints foot

    Parameters
    ----------
    Nothing

    Returns
    -------
    Nothing

    """

    print "  [Done]                                                                             "


def progress(title=None, fold=None, percentage=None, note=None, label=None):
    """Prints progress line

    Parameters
    ----------
    title : str or None
        Title

    fold : int > 0 [scalar] or None
        Fold number

    percentage : float [0-1] or None
        Progress percentage.

    note : str or None
        Note

    label : str or None
        Label

    Returns
    -------
    Nothing

    """

    if title is not None and fold is not None and percentage is not None and note is not None and label is None:
        print "  {:2s} {:20s} fold[{:1d}] [{:3.0f}%] [{:20s}]                        \r".format(spinner.next(), title, fold,percentage * 100, note),
    elif title is not None and fold is not None and percentage is None and note is not None and label is None:
        print "  {:2s} {:20s} fold[{:1d}]        [{:20s}]                     \r".format(spinner.next(), title, fold, note),
    elif title is not None and fold is None and percentage is not None and note is not None and label is None:
        print "  {:2s} {:20s} [{:3.0f}%] [{:20s}]                          \r".format(spinner.next(), title, percentage * 100, note),
    elif title is not None and fold is None and percentage is not None and note is None and label is None:
        print "  {:2s} {:20s} [{:3.0f}%]                                   \r".format(spinner.next(), title, percentage * 100),
    elif title is not None and fold is None and percentage is None and note is not None and label is None:
        print "  {:2s} {:20s} [{:20s}]                                    \r".format(spinner.next(), title, note),
    elif title is not None and fold is not None and percentage is not None and note is not None and label is not None:
        print "  {:2s} {:20s} fold[{:1d}] [{:10s}] [{:3.0f}%] [{:20s}]                           \r".format(spinner.next(), title, fold, label, percentage * 100, note),
    elif title is not None and fold is not None and percentage is None and note is None and label is not None:
        print "  {:2s} {:20s} fold[{:1d}] [{:10s}]                                               \r".format(spinner.next(), title, fold, label),

    sys.stdout.flush()
