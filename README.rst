torchvision
===========

Cloned verion of torchvision. Add 'register_custom_op_symbolic' for deform_conv2d


Requirements
============

.. code:: bash

    torchvision==0.5.0
    torch==1.4.0

Installation
============

From source:

.. code:: bash

    python setup.py install

By default, GPU support is built if CUDA is found and ``torch.cuda.is_available()`` is true.
It's possible to force building GPU support by setting ``FORCE_CUDA=1`` environment variable,
which is useful when building a docker image.


History
============
- 2020/11/23: Register custom operator for deform_conv2d
