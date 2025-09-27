"""
Comprehensive security and access control system for home automation.
Provides authentication, authorization, audit logging, and security monitoring.
"""

import asyncio
import logging
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import ipaddress
from pathlib import Path

logger = logging.getLogger(__name__)

class UserRole(Enum):
    GUEST = "guest"
    USER = "user"
    ADMIN = "admin"
    SYSTEM = "system"

class Permission(Enum):
    READ_DEVICES = "read_devices"
    CONTROL_DEVICES = "control_devices"
    MANAGE_AUTOMATIONS = "manage_automations"
    VIEW_HISTORY = "view_history"
    MANAGE_USERS = "manage_users"
    SYSTEM_CONFIG = "system_config"
    SECURITY_ADMIN = "security_admin"

class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EventType(Enum):
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    DEVICE_ACCESS = "device_access"
    PERMISSION_DENIED = "permission_denied"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_CHANGE = "system_change"

@dataclass
class User:
    user_id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    permissions: Set[Permission]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions

    def is_locked(self) -> bool:
        return self.locked_until and self.locked_until > datetime.now()

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        data = {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'role': self.role.value,
            'permissions': [p.value for p in self.permissions],
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'metadata': self.metadata
        }

        if include_sensitive:
            data.update({
                'failed_login_attempts': self.failed_login_attempts,
                'locked_until': self.locked_until.isoformat() if self.locked_until else None
            })

        return data

@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    last_activity: Optional[datetime] = None

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'is_active': self.is_active,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None
        }

@dataclass
class SecurityEvent:
    event_id: str
    event_type: EventType
    user_id: Optional[str]
    ip_address: str
    timestamp: datetime
    details: Dict[str, Any]
    severity: SecurityLevel = SecurityLevel.LOW

    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'user_id': self.user_id,
            'ip_address': self.ip_address,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details,
            'severity': self.severity.value
        }

@dataclass
class SecurityConfig:
    jwt_secret: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    session_timeout_hours: int = 24
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    password_min_length: int = 8
    require_special_chars: bool = True
    allowed_ip_ranges: List[str] = field(default_factory=list)
    blocked_ip_addresses: Set[str] = field(default_factory=set)
    enable_audit_logging: bool = True
    max_concurrent_sessions: int = 5

class SecurityManager:
    """
    Comprehensive security and access control system.
    Handles authentication, authorization, session management, and security monitoring.
    """

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()

        # User management
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}

        # Security monitoring
        self.security_events: List[SecurityEvent] = []
        self.failed_login_attempts: Dict[str, List[datetime]] = {}
        self.blocked_ips: Set[str] = set()

        # Role-based permissions
        self.role_permissions = self._initialize_role_permissions()

        # Security callbacks
        self.security_callbacks: List[Callable[[SecurityEvent], None]] = []

        # Initialize default admin user
        self._create_default_admin()

    def _initialize_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Initialize default role-permission mappings"""
        return {
            UserRole.GUEST: {
                Permission.READ_DEVICES,
                Permission.VIEW_HISTORY
            },
            UserRole.USER: {
                Permission.READ_DEVICES,
                Permission.CONTROL_DEVICES,
                Permission.VIEW_HISTORY
            },
            UserRole.ADMIN: {
                Permission.READ_DEVICES,
                Permission.CONTROL_DEVICES,
                Permission.MANAGE_AUTOMATIONS,
                Permission.VIEW_HISTORY,
                Permission.MANAGE_USERS,
                Permission.SYSTEM_CONFIG
            },
            UserRole.SYSTEM: set(Permission)  # System has all permissions
        }

    def _create_default_admin(self) -> None:
        """Create default admin user if none exists"""
        if not any(user.role == UserRole.ADMIN for user in self.users.values()):
            default_password = "admin123"  # Should be changed immediately
            admin_user = User(
                user_id="admin_001",
                username="admin",
                email="admin@localhost",
                password_hash=self._hash_password(default_password),
                role=UserRole.ADMIN,
                permissions=self.role_permissions[UserRole.ADMIN],
                created_at=datetime.now()
            )
            self.users[admin_user.user_id] = admin_user
            logger.warning("Created default admin user - please change password immediately")

    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256 with salt"""
        salt = secrets.token_hex(32)
        hash_obj = hashlib.sha256((password + salt).encode())
        return f"{salt}:{hash_obj.hexdigest()}"

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        try:
            salt, hash_value = password_hash.split(':', 1)
            hash_obj = hashlib.sha256((password + salt).encode())
            return hash_obj.hexdigest() == hash_value
        except ValueError:
            return False

    def _validate_password_strength(self, password: str) -> List[str]:
        """Validate password meets security requirements"""
        errors = []

        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters")

        if self.config.require_special_chars:
            if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
                errors.append("Password must contain at least one special character")

            if not any(c.isupper() for c in password):
                errors.append("Password must contain at least one uppercase letter")

            if not any(c.islower() for c in password):
                errors.append("Password must contain at least one lowercase letter")

            if not any(c.isdigit() for c in password):
                errors.append("Password must contain at least one digit")

        return errors

    def _is_ip_allowed(self, ip_address: str) -> bool:
        """Check if IP address is allowed"""
        if ip_address in self.blocked_ips or ip_address in self.config.blocked_ip_addresses:
            return False

        if not self.config.allowed_ip_ranges:
            return True  # No restrictions if no ranges specified

        try:
            ip = ipaddress.ip_address(ip_address)
            for range_str in self.config.allowed_ip_ranges:
                if ip in ipaddress.ip_network(range_str, strict=False):
                    return True
        except ValueError:
            return False

        return False

    def _record_security_event(self, event_type: EventType, user_id: Optional[str],
                             ip_address: str, details: Dict[str, Any],
                             severity: SecurityLevel = SecurityLevel.LOW) -> None:
        """Record a security event"""
        event = SecurityEvent(
            event_id=secrets.token_urlsafe(16),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            timestamp=datetime.now(),
            details=details,
            severity=severity
        )

        self.security_events.append(event)

        # Limit event history
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-5000:]

        # Call security callbacks
        for callback in self.security_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in security callback: {e}")

        if self.config.enable_audit_logging:
            logger.info(f"Security event: {event_type.value} - {details}")

    async def create_user(self, username: str, email: str, password: str,
                         role: UserRole, permissions: Optional[Set[Permission]] = None,
                         creator_user_id: Optional[str] = None) -> User:
        """Create a new user"""
        # Validate password
        password_errors = self._validate_password_strength(password)
        if password_errors:
            raise ValueError(f"Password validation failed: {', '.join(password_errors)}")

        # Check if username already exists
        if any(user.username == username for user in self.users.values()):
            raise ValueError(f"Username '{username}' already exists")

        # Use role-based permissions if not specified
        if permissions is None:
            permissions = self.role_permissions.get(role, set())

        user = User(
            user_id=secrets.token_urlsafe(16),
            username=username,
            email=email,
            password_hash=self._hash_password(password),
            role=role,
            permissions=permissions,
            created_at=datetime.now()
        )

        self.users[user.user_id] = user

        self._record_security_event(
            EventType.SYSTEM_CHANGE,
            creator_user_id,
            "system",
            {"action": "create_user", "target_user": user.username, "role": role.value}
        )

        logger.info(f"Created user: {username} with role {role.value}")
        return user

    async def authenticate_user(self, username: str, password: str,
                              ip_address: str, user_agent: str) -> Optional[Session]:
        """Authenticate a user and create a session"""
        # Check IP restrictions
        if not self._is_ip_allowed(ip_address):
            self._record_security_event(
                EventType.SECURITY_VIOLATION,
                None,
                ip_address,
                {"reason": "blocked_ip", "username": username},
                SecurityLevel.HIGH
            )
            raise PermissionError("Access denied from this IP address")

        # Find user
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break

        if not user or not user.is_active:
            self._record_security_event(
                EventType.LOGIN_FAILURE,
                None,
                ip_address,
                {"username": username, "reason": "invalid_user"}
            )
            return None

        # Check if user is locked
        if user.is_locked():
            self._record_security_event(
                EventType.LOGIN_FAILURE,
                user.user_id,
                ip_address,
                {"username": username, "reason": "account_locked"},
                SecurityLevel.MEDIUM
            )
            raise PermissionError("Account is temporarily locked")

        # Verify password
        if not self._verify_password(password, user.password_hash):
            user.failed_login_attempts += 1

            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.config.max_login_attempts:
                user.locked_until = datetime.now() + timedelta(minutes=self.config.lockout_duration_minutes)

            self._record_security_event(
                EventType.LOGIN_FAILURE,
                user.user_id,
                ip_address,
                {"username": username, "reason": "invalid_password", "attempts": user.failed_login_attempts}
            )
            return None

        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()

        # Check concurrent session limit
        active_sessions = self._get_user_sessions(user.user_id)
        if len(active_sessions) >= self.config.max_concurrent_sessions:
            # Remove oldest session
            oldest_session = min(active_sessions, key=lambda s: s.created_at)
            await self.invalidate_session(oldest_session.session_id)

        # Create session
        session = Session(
            session_id=secrets.token_urlsafe(32),
            user_id=user.user_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=self.config.session_timeout_hours),
            ip_address=ip_address,
            user_agent=user_agent,
            last_activity=datetime.now()
        )

        self.sessions[session.session_id] = session

        self._record_security_event(
            EventType.LOGIN_SUCCESS,
            user.user_id,
            ip_address,
            {"username": username, "session_id": session.session_id}
        )

        logger.info(f"User {username} authenticated successfully")
        return session

    async def validate_session(self, session_id: str, ip_address: str) -> Optional[User]:
        """Validate a session and return the associated user"""
        session = self.sessions.get(session_id)
        if not session or not session.is_active or session.is_expired():
            return None

        # Check IP consistency (optional security measure)
        if session.ip_address != ip_address:
            self._record_security_event(
                EventType.SECURITY_VIOLATION,
                session.user_id,
                ip_address,
                {"reason": "ip_mismatch", "session_ip": session.ip_address, "request_ip": ip_address},
                SecurityLevel.MEDIUM
            )

        # Update last activity
        session.last_activity = datetime.now()

        user = self.users.get(session.user_id)
        if not user or not user.is_active:
            await self.invalidate_session(session_id)
            return None

        return user

    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session"""
        session = self.sessions.get(session_id)
        if session:
            session.is_active = False

            self._record_security_event(
                EventType.LOGOUT,
                session.user_id,
                session.ip_address,
                {"session_id": session_id}
            )

            logger.info(f"Session {session_id} invalidated")
            return True

        return False

    async def check_permission(self, session_id: str, permission: Permission,
                             ip_address: str, resource: str = None) -> bool:
        """Check if a session has a specific permission"""
        user = await self.validate_session(session_id, ip_address)
        if not user:
            return False

        has_permission = user.has_permission(permission)

        if not has_permission:
            self._record_security_event(
                EventType.PERMISSION_DENIED,
                user.user_id,
                ip_address,
                {"permission": permission.value, "resource": resource},
                SecurityLevel.MEDIUM
            )

        return has_permission

    async def change_password(self, user_id: str, old_password: str,
                            new_password: str, ip_address: str) -> bool:
        """Change a user's password"""
        user = self.users.get(user_id)
        if not user:
            return False

        # Verify old password
        if not self._verify_password(old_password, user.password_hash):
            self._record_security_event(
                EventType.SECURITY_VIOLATION,
                user_id,
                ip_address,
                {"action": "change_password", "reason": "invalid_old_password"},
                SecurityLevel.MEDIUM
            )
            return False

        # Validate new password
        password_errors = self._validate_password_strength(new_password)
        if password_errors:
            raise ValueError(f"Password validation failed: {', '.join(password_errors)}")

        # Update password
        user.password_hash = self._hash_password(new_password)

        self._record_security_event(
            EventType.SYSTEM_CHANGE,
            user_id,
            ip_address,
            {"action": "password_changed"}
        )

        logger.info(f"Password changed for user {user.username}")
        return True

    async def update_user_permissions(self, target_user_id: str, permissions: Set[Permission],
                                    admin_user_id: str, ip_address: str) -> bool:
        """Update a user's permissions"""
        target_user = self.users.get(target_user_id)
        admin_user = self.users.get(admin_user_id)

        if not target_user or not admin_user:
            return False

        # Check admin permissions
        if not admin_user.has_permission(Permission.MANAGE_USERS):
            self._record_security_event(
                EventType.PERMISSION_DENIED,
                admin_user_id,
                ip_address,
                {"action": "update_permissions", "target_user": target_user_id},
                SecurityLevel.HIGH
            )
            return False

        old_permissions = target_user.permissions.copy()
        target_user.permissions = permissions

        self._record_security_event(
            EventType.SYSTEM_CHANGE,
            admin_user_id,
            ip_address,
            {
                "action": "update_permissions",
                "target_user": target_user.username,
                "old_permissions": [p.value for p in old_permissions],
                "new_permissions": [p.value for p in permissions]
            }
        )

        logger.info(f"Updated permissions for user {target_user.username}")
        return True

    def get_security_events(self, user_id: Optional[str] = None,
                          event_type: Optional[EventType] = None,
                          hours: int = 24) -> List[SecurityEvent]:
        """Get security events with optional filtering"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        events = [e for e in self.security_events if e.timestamp > cutoff_time]

        if user_id:
            events = [e for e in events if e.user_id == user_id]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return sorted(events, key=lambda e: e.timestamp, reverse=True)

    def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all sessions for a user"""
        return self._get_user_sessions(user_id)

    def _get_user_sessions(self, user_id: str) -> List[Session]:
        """Internal method to get user sessions"""
        return [s for s in self.sessions.values() if s.user_id == user_id and s.is_active and not s.is_expired()]

    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)

        recent_events = [e for e in self.security_events if e.timestamp > last_24h]

        return {
            'total_users': len(self.users),
            'active_users': len([u for u in self.users.values() if u.is_active]),
            'locked_users': len([u for u in self.users.values() if u.is_locked()]),
            'active_sessions': len([s for s in self.sessions.values() if s.is_active and not s.is_expired()]),
            'events_last_24h': len(recent_events),
            'login_failures_last_24h': len([e for e in recent_events if e.event_type == EventType.LOGIN_FAILURE]),
            'security_violations_last_24h': len([e for e in recent_events if e.event_type == EventType.SECURITY_VIOLATION]),
            'blocked_ips': len(self.blocked_ips)
        }

    def add_security_callback(self, callback: Callable[[SecurityEvent], None]) -> None:
        """Add a callback for security events"""
        self.security_callbacks.append(callback)

    async def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions"""
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.is_expired()
        ]

        for session_id in expired_sessions:
            await self.invalidate_session(session_id)

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

# JWT Token utilities for API authentication

class JWTTokenManager:
    """JWT token manager for API authentication"""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def create_token(self, user_id: str, permissions: List[str], expires_hours: int = 24) -> str:
        """Create a JWT token"""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': datetime.utcnow() + timedelta(hours=expires_hours),
            'iat': datetime.utcnow()
        }

        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None