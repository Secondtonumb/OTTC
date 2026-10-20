"""Read-only PBS status using the installed client library (qstat fallback)."""
import ctypes as c
import json
import re
import sys


class Attr(c.Structure):
    pass


Attr._fields_ = [('next', c.POINTER(Attr)), ('name', c.c_char_p),
                ('resource', c.c_char_p), ('value', c.c_char_p), ('op', c.c_int)]


class Status(c.Structure):
    pass


Status._fields_ = [('next', c.POINTER(Status)), ('name', c.c_char_p),
                  ('attribs', c.POINTER(Attr)), ('text', c.c_char_p), ('objtype', c.c_int)]


def main():
    if not sys.argv[1:] or not all(re.fullmatch(r'\d+\.opbs', x) for x in sys.argv[1:]):
        raise SystemExit('Usage: pbs_status.py JOBID.opbs [...]')
    pbs = c.CDLL('/opt/pbs/lib/libpbs.so', mode=c.RTLD_GLOBAL)
    pbs.pbs_connect.argtypes = [c.c_char_p]; pbs.pbs_connect.restype = c.c_int
    pbs.pbs_statjob.argtypes = [c.c_int, c.c_char_p, c.POINTER(Attr), c.c_char_p]
    pbs.pbs_statjob.restype = c.POINTER(Status)
    pbs.pbs_statfree.argtypes = [c.POINTER(Status)]
    pbs.pbs_disconnect.argtypes = [c.c_int]
    con = pbs.pbs_connect(b'opbs')
    if con < 0:
        raise SystemExit('PBS connection failed')
    try:
        for job in sys.argv[1:]:
            s = pbs.pbs_statjob(con, job.encode(), None, b'x')
            result = {'job': job}
            if s:
                a = s.contents.attribs
                while a:
                    q = a.contents; name = q.name.decode()
                    resource = q.resource.decode() if q.resource else ''
                    if name in ('job_state', 'Exit_status', 'comment', 'exec_host') or name == 'resources_used' and resource in ('walltime','mem','ncpus','cput'):
                        result[name + ('/' + resource if resource else '')] = q.value.decode()
                    a = q.next
                pbs.pbs_statfree(s)
            else:
                result['status'] = 'not returned by PBS'
            print(json.dumps(result))
    finally:
        pbs.pbs_disconnect(con)


if __name__ == '__main__': main()
