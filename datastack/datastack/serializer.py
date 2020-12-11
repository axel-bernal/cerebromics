""":mod:`datastack.serializer` defines classes and methods to support
serializing objects to and deserializing objects from stable storage. This
module improves upon Python's :mod:`pickle` module by requiring that
serializable classes specify which internal data values to store and load,
and by removing the implicit dependencies on module code.
"""

import logging
import numpy as np
import six

from abc import ABCMeta, abstractmethod, abstractproperty

_logger = logging.getLogger(__name__)


_serializer_version_tag = '__serializer__'
_obj_class_tag = '__class__'
_obj_version_tag = '__version__'
_nested_values_tag = '__nested__'


def version():
    return 1


def _has_serializer_tags(data):
    return set([_serializer_version_tag, _obj_class_tag, _obj_version_tag, _nested_values_tag]) <= set(data.keys())


def _is_primitve(obj):
    """Check if an object is a primitive. This is a bit of a hack - we
    assume that an object must be a 'primitive' type if it has no
    dictionary attribute or if it is an instance of the `type` object.
    """
    return not hasattr(obj, '__dict__') or isinstance(obj, type)


def _is_serializable(obj):
    return issubclass(obj.__class__, Serializable)


def _serialize(obj, prefix=''):
    if not _is_serializable(obj):
        raise RuntimeError('Unable to serialize class {}'.format(obj.__class__))
    data = obj._do_serialize()
    for k in data.keys():
        # If we find a serializable data value in the dict, recursively
        # serialize the value.
        v = data[k]
        if _is_serializable(v):
            _logger.debug('Recursively serializing obj of type {} and re-keying...'.format(type(v)))
            data[_nested_values_tag].append(k)
            # Add a prefix to the subvalues in the serialiable data value.
            v_data = _serialize(v, prefix + '{}.'.format(k))
            for v_data_k in v_data:
                data[v_data_k] = v_data[v_data_k]
            del data[k]
        elif _is_primitve(v):
            if prefix != '':
                _logger.debug('Not recursively serializing obj of type {}, but re-keying...'.format(type(v)))
                data['{}{}'.format(prefix, k)] = data[k]
                del data[k]
            else:
                _logger.debug('Not recursively serializing obj of type {}'.format(type(v)))
        else:
            raise RuntimeError('Unable to serialize class {}'.format(obj.__class__))
    return data


def _deserialize(data):
    if not _has_serializer_tags(data):
        raise RuntimeError('Data dictionary does not contain a valid serialized object')
    # We do the reverse of _serialize() - that is, we have to deserialize
    # nested values before we finally deserialize the object.
    for nested_k in data[_nested_values_tag]:
        nested_prefix = '{}.'.format(nested_k)
        # Create a new data dictionary containing just the nested
        # serialized data
        nested_data = {}
        for k in data.keys():
            if k.startswith(nested_prefix):
                nested_data[k.replace(nested_prefix, '')] = data[k]
                del data[k]
        if len(nested_data) == 0:
            _logger.warning('Unable to find data for nested value {}'.format(nested_k))
        # Deserialized the nested serialized data
        data[nested_k] = _deserialize(nested_data)
    obj = data[_obj_class_tag]()
    obj.deserialize(data)
    return obj


class Serializable(six.with_metaclass(ABCMeta)):
    """:class:`Serializable` is an abstract base class for all serialiable
    classes. Classes that inherit from :class:`Serializable` must implement
    methods for serializing and deserializing their internal data, and
    declare a version number for their internal data representation. Note
    that subclasses of :class:`Serializable` **must** implement the a
    constructor that either takes no arguments or includes default values
    for all arguments.
    """

    @abstractproperty
    def serialized_version(self):
        """Return the version number of the internal data representation.
        """
        raise NotImplementedError

    def get_serialized_version(self, data):
        """Get the version number of an internal data representation from a
        data dictionary.
        """
        return data[_serializer_version_tag]

    @abstractmethod
    def serialize(self, data):
        """Populate a dictionary of internal data to serialize. Values in
        the dictionary must either be `primitive` Python types (such as
        integers or strings) or be classes that inherit from
        :class:`Serializable` and implement all of the abstract methods.
        """
        raise NotImplementedError

    @abstractmethod
    def deserialize(self, data):
        """Given a data dictionary, restore the internal data to an
        instance.

        :param data: A data dictionary containing (deserialized) internal
        data.
        """
        pass

    def _do_serialize(self):
        """Internal helper method to create the base data dictionary with
        header metadata.
        """
        data = {}
        data[_serializer_version_tag] = version()
        data[_obj_class_tag] = self.__class__
        data[_obj_version_tag] = self.serialized_version
        data[_nested_values_tag] = []
        return self.serialize(data)


def save(obj, objfile):
    """Save a serializable object to a file.

    :param obj: The object to serialize.
    :type obj: Subclass of :class:`datastack.serializer.Serializable`.
    :param objfile: The file to which to store the serialized object.
    :type objfile: A filename or a file handle
    """
    _logger.debug('Serializing obj of type {}...'.format(obj.__class__))
    data = _serialize(obj)
    np.savez(file=objfile, **data)


def load(objfile):
    """Load a serialized object to a file.

    :param objfile: The file from which to load the serialized object.
    :type objfile: A filename or a file handle
    """
    data = np.load(objfile)
    data_cooked = {}
    for k in data.keys():
        # np.save stores singleton values in zero-dimension arrays (!), so
        # we need to force a list to recover such values.
        if data[k].shape == ():
            data_cooked[k] = data[k].tolist()
        else:
            data_cooked[k] = data[k]
    try:
        return _deserialize(data_cooked)
    except:
        raise RuntimeError('Object file {} does not contain a valid serialized object'.format(objfile))
