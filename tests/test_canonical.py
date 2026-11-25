import pytest
from src.data.canonical import canonicalise_url, is_s_strict, is_s_ctrl, normalise_nfc

def test_canonicalise_url():
    u1 = "HTTPS://WWW.Example.COM/foo/bar/"
    assert canonicalise_url(u1) == "https://example.com/foo/bar"
    
    u2 = "https://example.com/"
    assert canonicalise_url(u2) == "https://example.com"
    
    u3 = "http://example.com:8080/test?a=1#frag"
    assert canonicalise_url(u3) == "http://example.com:8080/test?a=1#frag"

def test_s_strict_predicate():
    assert is_s_strict("https://example.com") is True
    assert is_s_strict("https://sub.domain.org") is True
    assert is_s_strict("http://example.com") is False # not https
    assert is_s_strict("https://example.com/path") is False # has path
    assert is_s_strict("https://example.com?query=1") is False # has query
    assert is_s_strict("https://example.com#frag") is False # has fragment
    assert is_s_strict("https://user:pass@example.com") is False # userinfo
    assert is_s_strict("https://example.com:8443") is False # non-standard port
