def memoize(func):
    """
    Декоратор для кешування результатів функцій, щоб уникнути повторних обчислень.
    """
    cache = {}
    
    def wrapper(*args, **kwargs):
        # Створюємо унікальний ключ на основі аргументів
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    # Додаємо функцію для очищення кешу
    wrapper.clear_cache = lambda: cache.clear()
    return wrapper