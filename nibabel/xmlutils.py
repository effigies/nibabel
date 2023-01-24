# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Thin layer around xml.etree.ElementTree, to abstract nibabel xml support.
"""
from __future__ import annotations

import io
import typing as ty
from xml.etree.ElementTree import Element, SubElement, tostring  # noqa
from xml.parsers.expat import ParserCreate, XMLParserType

from .filebasedimages import FileBasedHeader

if ty.TYPE_CHECKING:  # pragma: no cover
    from _typeshed import SupportsRead


class XmlSerializable:
    """Basic interface for serializing an object to xml"""

    def _to_xml_element(self) -> Element | None:
        """Output should be a xml.etree.ElementTree.Element"""
        raise NotImplementedError  # pragma: no cover

    def to_xml(self, enc: str = 'utf-8') -> bytes:
        """Output should be an xml string with the given encoding.
        (default: utf-8)"""
        ele = self._to_xml_element()
        return b'' if ele is None else tostring(ele, enc)


class XmlBasedHeader(FileBasedHeader, XmlSerializable):
    """Basic wrapper around FileBasedHeader and XmlSerializable."""


class XmlParser:
    """Base class for defining how to parse xml-based image snippets.

    Image-specific parsers should define:
        StartElementHandler
        EndElementHandler
        CharacterDataHandler
    """

    HANDLER_NAMES = ['StartElementHandler', 'EndElementHandler', 'CharacterDataHandler']

    def __init__(
        self,
        encoding: str = 'utf-8',
        buffer_size: int | None = 35000000,
        verbose: int = 0,
    ):
        """
        Parameters
        ----------
        encoding : str
            string containing xml document

        buffer_size: None or int, optional
            size of read buffer. None uses default buffer_size
            from xml.parsers.expat.

        verbose : int, optional
            amount of output during parsing (0=silent, by default).
        """
        self.encoding = encoding
        self.buffer_size = buffer_size
        self.verbose = verbose
        self.fname = None  # set on calls to parse

    def _create_parser(self) -> XMLParserType:
        """Internal function that allows subclasses to mess
        with the underlying parser, if desired."""

        parser = ParserCreate(encoding=self.encoding)  # from xml package
        parser.buffer_text = True
        if self.buffer_size is not None:
            parser.buffer_size = self.buffer_size
        return parser

    def parse(
        self,
        string: bytes | None = None,
        fname: str | None = None,
        fptr: SupportsRead[bytes] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        string : bytes
            string (as a bytes object) containing xml document

        fname : str
            file name of an xml document.

        fptr : file pointer
            open file pointer to an xml documents
        """
        if int(string is not None) + int(fptr is not None) + int(fname is not None) != 1:
            raise ValueError('Exactly one of fptr, fname, string must be specified.')

        if fname is not None:
            fptr = open(fname, 'rb')
        elif string is not None:
            fptr = io.BytesIO(string)
        else:
            # For type narrowing
            assert fptr is not None

        # store the name of the xml file in case it is needed during parsing
        self.fname = getattr(fptr, 'name', None)
        parser = self._create_parser()
        for name in self.HANDLER_NAMES:
            setattr(parser, name, getattr(self, name))
        parser.ParseFile(fptr)

    def StartElementHandler(self, name: str, attrs: dict[str, str]) -> None:
        raise NotImplementedError  # pragma: no cover

    def EndElementHandler(self, name: str) -> None:
        raise NotImplementedError  # pragma: no cover

    def CharacterDataHandler(self, data: str) -> None:
        raise NotImplementedError  # pragma: no cover
