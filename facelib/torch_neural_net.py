import atexit
import binascii
from subprocess import Popen, PIPE
import os
import os.path
import sys

import numpy as np
import cv2

myDir = os.path.dirname(os.path.realpath(__file__))

os.environ['TERM'] = 'linux'


class TorchNeuralNet:

    defaultModel = os.path.join(myDir, '..', 'models', 'fcn.t7')

    def __init__(self, model=defaultModel, imgDim=96, cuda=False):

        assert model is not None
        assert imgDim is not None
        assert cuda is not None

        self.cmd = ['/usr/bin/env', 'th', os.path.join(myDir, 'forward.lua'),
                    '-model', model, '-imgDim', str(imgDim)]
        if cuda:
            self.cmd.append('-cuda')
        self.p = Popen(self.cmd, stdin=PIPE, stdout=PIPE, bufsize=0, universal_newlines=True)

        def exitHandler():
            if self.p.poll() is None:
                self.p.kill()
        atexit.register(exitHandler)

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        if self.p.poll() is None:
            self.p.kill()


    def __del__(self):

        if self.p.poll() is None:
            self.p.kill()

    def forwardPath(self, imgPath):

        assert imgPath is not None

        rc = self.p.poll()
        if rc is not None and rc != 0:
            raise Exception("""




Diagnostic information:

cmd: {}

============

stdout: {}
""".format(self.cmd, self.p.stdout.read()))

        self.p.stdin.write(imgPath + '\n')
        output = self.p.stdout.readline()
        try:
            rep = [float(x) for x in output.strip().split(',')]
            rep = np.array(rep)
            return rep
        except Exception as e:
            self.p.kill()
            stdout, stderr = self.p.communicate()
            print("""


Error getting result from Torch subprocess.

Line read: {}

Exception:

{}

============

stdout: {}
""".format(output, str(e), stdout))
            sys.exit(-1)

    def forward(self, rgbImg):
       
        assert rgbImg is not None

        t = '/tmp/openface-torchwrap-{}.png'.format(
            binascii.b2a_hex(os.urandom(8)))
        bgrImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2BGR)
        cv2.imwrite(t, bgrImg)
        rep = self.forwardPath(t)
        os.remove(t)
        return rep
