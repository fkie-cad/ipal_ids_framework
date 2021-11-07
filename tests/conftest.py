from subprocess import Popen, PIPE

METAIDS = "./ipal-iids"

########################
# Helper methods
########################


def metaids(args):
    p = Popen([METAIDS] + args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    return p.returncode, stdout, stderr


def file_eq(path, string):
    with open(path, "rb") as f:
        return string == f.read()
