# analytics/auth.py
from __future__ import annotations
import os
from typing import Dict, Any, Optional

# ВАЖНО: никаких обязательных импортов на модульном уровне.
# Всё импортируем лениво внутри функций, чтобы не падать при первом старте.

_DEFAULT_USERS_YAML_PLAINTEXT = """\
credentials:
  usernames:
    admin:
      email: admin@example.com
      name: Administrator
      password: admin123        # plain-text; app.py захеширует при старте
      role: admin
cookie:
  name: analytics_app
  key: change_me_please
  expiry_days: 14
preauthorized:
  emails: []
"""


def ensure_users_file(path: str) -> None:
    """Создаёт config/users.yaml, если его нет. Не требует внешних пакетов."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(_DEFAULT_USERS_YAML_PLAINTEXT)


def load_users(path: str) -> Dict[str, Any]:
    import yaml  # ленивый импорт
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_users(path: str, data: Dict[str, Any]) -> None:
    import yaml  # ленивый импорт
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _hash_password(p: str) -> str:
    """Пытаемся захешировать пароли без streamlit-authenticator (через bcrypt).
    Если ни один вариант недоступен — вернём plain-text (app.py захеширует при следующем старте)."""
    try:
        import bcrypt
        return bcrypt.hashpw(p.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    except Exception:
        try:
            from streamlit_authenticator import Hasher
            # поддержка новых и старых версий
            if hasattr(Hasher, "generate"):
                try:
                    return Hasher().generate([p])[0]
                except TypeError:
                    return Hasher([p]).generate()[0]
            if hasattr(Hasher, "hash_passwords"):
                return Hasher().hash_passwords([p])[0]
        except Exception:
            pass
    return p  # fallback: plain (потом захеширует app.py)


def add_user(data: Dict[str, Any], username: str, name: str, email: str, password_plain: str, role: str = "user") -> None:
    creds = data.setdefault("credentials", {}).setdefault("usernames", {})
    if username in creds:
        raise ValueError("Пользователь с таким username уже существует")
    creds[username] = {"email": email, "name": name,
                       "password": _hash_password(password_plain), "role": role}


def update_user_password(data: Dict[str, Any], username: str, new_password_plain: str) -> None:
    creds = data.get("credentials", {}).get("usernames", {})
    if username not in creds:
        raise ValueError("Пользователь не найден")
    creds[username]["password"] = _hash_password(new_password_plain)


def delete_user(data: Dict[str, Any], username: str) -> None:
    creds = data.get("credentials", {}).get("usernames", {})
    if username not in creds:
        raise ValueError("Пользователь не найден")
    del creds[username]


def set_user_role(data: Dict[str, Any], username: str, role: str) -> None:
    creds = data.get("credentials", {}).get("usernames", {})
    if username not in creds:
        raise ValueError("Пользователь не найден")
    creds[username]["role"] = role


def get_user_role(data: Dict[str, Any], username: str) -> Optional[str]:
    creds = data.get("credentials", {}).get("usernames", {})
    return creds.get(username, {}).get("role")
