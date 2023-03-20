from src.utils import classproperty

class Parent:
    positive = 2
    @classproperty
    def negative(cls):
        return cls.positive * -1

class Child(Parent):
    positive = 1

def test_classproperty():
    assert Parent.positive == 2
    assert Parent.negative == -2

def test_classproperty_inheritance():
    assert Child.positive == 1
    assert Child.negative == -1