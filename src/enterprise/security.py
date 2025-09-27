"""
Advanced Security Framework
Enterprise-grade authentication, authorization, and encryption
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import secrets
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union
from uuid import uuid4

import bcrypt
import jwt
from cryptography.fernet import Fernet
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class AuthMethod(str, Enum):
    """Authentication methods"""
    PASSWORD = "password"
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    TWO_FACTOR = "two_factor"
    SSO = "sso"


class UserRole(str, Enum):
    """User roles"""
    SUPER_ADMIN = "super_admin"
    TENANT_ADMIN = "tenant_admin"
    USER_MANAGER = "user_manager"
    AGENT_OPERATOR = "agent_operator"
    ANALYST = "analyst"
    USER = "user"
    GUEST = "guest"
    API_CLIENT = "api_client"


class Permission(str, Enum):
    """System permissions"""
    # Tenant management
    TENANT_CREATE = "tenant:create"
    TENANT_READ = "tenant:read"
    TENANT_UPDATE = "tenant:update"
    TENANT_DELETE = "tenant:delete"
    TENANT_ADMIN = "tenant:admin"

    # User management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    USER_IMPERSONATE = "user:impersonate"

    # Agent operations
    AGENT_CREATE = "agent:create"
    AGENT_READ = "agent:read"
    AGENT_UPDATE = "agent:update"
    AGENT_DELETE = "agent:delete"
    AGENT_EXECUTE = "agent:execute"

    # Workflow operations
    WORKFLOW_CREATE = "workflow:create"
    WORKFLOW_READ = "workflow:read"
    WORKFLOW_UPDATE = "workflow:update"
    WORKFLOW_DELETE = "workflow:delete"
    WORKFLOW_EXECUTE = "workflow:execute"

    # Memory operations
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    MEMORY_DELETE = "memory:delete"
    MEMORY_ADMIN = "memory:admin"

    # Analytics and monitoring
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_EXPORT = "analytics:export"
    MONITORING_READ = "monitoring:read"
    MONITORING_ADMIN = "monitoring:admin"

    # System administration
    SYSTEM_CONFIG = "system:config"
    SYSTEM_LOGS = "system:logs"
    SYSTEM_HEALTH = "system:health"
    SYSTEM_ADMIN = "system:admin"

    # API access
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"


class SecurityEvent(BaseModel):
    """Security audit event"""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    severity: str  # "low", "medium", "high", "critical"
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: str  # "success", "failure", "blocked"
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        extra = "forbid"


class TwoFactorAuth(BaseModel):
    """Two-factor authentication data"""
    secret: str
    backup_codes: List[str] = Field(default_factory=list)
    enabled: bool = False
    last_used: Optional[datetime] = None

    class Config:
        extra = "forbid"


class APIKey(BaseModel):
    """API key configuration"""
    key_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    key_hash: str  # Hashed version of the key
    permissions: List[Permission] = Field(default_factory=list)
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

    class Config:
        extra = "forbid"

    def is_valid(self) -> bool:
        """Check if API key is valid"""
        if not self.is_active:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True


class User(BaseModel):
    """User entity with security features"""
    user_id: str = Field(default_factory=lambda: str(uuid4()))
    username: str
    email: str
    password_hash: Optional[str] = None  # For password auth
    full_name: str

    # Security
    roles: List[UserRole] = Field(default_factory=list)
    permissions: List[Permission] = Field(default_factory=list)
    two_factor: Optional[TwoFactorAuth] = None
    api_keys: List[APIKey] = Field(default_factory=list)

    # Tenant association
    tenant_id: Optional[str] = None
    is_tenant_admin: bool = False

    # OAuth/SSO
    oauth_providers: Dict[str, str] = Field(default_factory=dict)  # provider -> external_id
    sso_enabled: bool = False

    # Security settings
    require_2fa: bool = False
    password_expires_at: Optional[datetime] = None
    account_locked: bool = False
    failed_login_attempts: int = 0
    last_login: Optional[datetime] = None
    last_password_change: Optional[datetime] = None

    # Audit
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"

    @validator('email')
    def validate_email(cls, v):
        """Basic email validation"""
        if '@' not in v or '.' not in v.split('@')[1]:
            raise ValueError('Invalid email format')
        return v.lower()

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission"""
        # Super admin has all permissions
        if UserRole.SUPER_ADMIN in self.roles:
            return True

        # Check direct permissions
        if permission in self.permissions:
            return True

        # Check role-based permissions
        role_permissions = self._get_role_permissions()
        return permission in role_permissions

    def has_role(self, role: UserRole) -> bool:
        """Check if user has specific role"""
        return role in self.roles

    def _get_role_permissions(self) -> Set[Permission]:
        """Get permissions from user roles"""
        permissions = set()

        for role in self.roles:
            if role == UserRole.SUPER_ADMIN:
                # Super admin gets all permissions
                permissions.update(Permission)
            elif role == UserRole.TENANT_ADMIN:
                permissions.update([
                    Permission.USER_CREATE, Permission.USER_READ, Permission.USER_UPDATE, Permission.USER_DELETE,
                    Permission.AGENT_CREATE, Permission.AGENT_READ, Permission.AGENT_UPDATE, Permission.AGENT_DELETE, Permission.AGENT_EXECUTE,
                    Permission.WORKFLOW_CREATE, Permission.WORKFLOW_READ, Permission.WORKFLOW_UPDATE, Permission.WORKFLOW_DELETE, Permission.WORKFLOW_EXECUTE,
                    Permission.MEMORY_READ, Permission.MEMORY_WRITE, Permission.MEMORY_DELETE,
                    Permission.ANALYTICS_READ, Permission.ANALYTICS_EXPORT,
                    Permission.MONITORING_READ,
                    Permission.API_READ, Permission.API_WRITE
                ])
            elif role == UserRole.USER_MANAGER:
                permissions.update([
                    Permission.USER_CREATE, Permission.USER_READ, Permission.USER_UPDATE,
                    Permission.ANALYTICS_READ
                ])
            elif role == UserRole.AGENT_OPERATOR:
                permissions.update([
                    Permission.AGENT_READ, Permission.AGENT_EXECUTE,
                    Permission.WORKFLOW_READ, Permission.WORKFLOW_EXECUTE,
                    Permission.MEMORY_READ, Permission.MEMORY_WRITE,
                    Permission.API_READ, Permission.API_WRITE
                ])
            elif role == UserRole.ANALYST:
                permissions.update([
                    Permission.AGENT_READ,
                    Permission.WORKFLOW_READ,
                    Permission.MEMORY_READ,
                    Permission.ANALYTICS_READ, Permission.ANALYTICS_EXPORT,
                    Permission.MONITORING_READ,
                    Permission.API_READ
                ])
            elif role == UserRole.USER:
                permissions.update([
                    Permission.AGENT_READ, Permission.AGENT_EXECUTE,
                    Permission.WORKFLOW_READ,
                    Permission.MEMORY_READ, Permission.MEMORY_WRITE,
                    Permission.API_READ
                ])
            elif role == UserRole.API_CLIENT:
                permissions.update([
                    Permission.API_READ, Permission.API_WRITE
                ])

        return permissions

    def is_account_valid(self) -> bool:
        """Check if account is valid for login"""
        if self.account_locked:
            return False
        if self.password_expires_at and datetime.utcnow() > self.password_expires_at:
            return False
        return True

    def update_last_login(self) -> None:
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        self.failed_login_attempts = 0  # Reset failed attempts on successful login

    def increment_failed_login(self, max_attempts: int = 5) -> bool:
        """Increment failed login attempts, return True if account should be locked"""
        self.failed_login_attempts += 1
        if self.failed_login_attempts >= max_attempts:
            self.account_locked = True
            return True
        return False


class JWTToken(BaseModel):
    """JWT token information"""
    token: str
    token_type: str = "bearer"
    expires_at: datetime
    user_id: str
    tenant_id: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)

    class Config:
        extra = "forbid"


class EncryptionManager:
    """Data encryption and decryption"""

    def __init__(self, key: Optional[bytes] = None):
        self.key = key or Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def encrypt(self, data: Union[str, bytes]) -> str:
        """Encrypt data and return base64 encoded string"""
        if isinstance(data, str):
            data = data.encode('utf-8')

        encrypted = self.cipher.encrypt(data)
        return base64.b64encode(encrypted).decode('utf-8')

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt base64 encoded string"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Invalid encrypted data")

    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary as JSON"""
        json_str = json.dumps(data)
        return self.encrypt(json_str)

    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt to dictionary"""
        json_str = self.decrypt(encrypted_data)
        return json.loads(json_str)


class SecurityManager:
    """Central security management system"""

    def __init__(self, secret_key: str, encryption_key: Optional[bytes] = None):
        self.secret_key = secret_key
        self.encryption_manager = EncryptionManager(encryption_key)

        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.security_events: List[SecurityEvent] = []

        self.jwt_algorithm = "HS256"
        self.jwt_expiry_hours = 24
        self.max_failed_logins = 5

    # User Management
    async def create_user(self,
                        username: str,
                        email: str,
                        password: str,
                        full_name: str,
                        roles: List[UserRole] = None,
                        tenant_id: Optional[str] = None,
                        created_by: Optional[str] = None) -> User:
        """Create a new user"""

        # Check if username/email already exists
        for user in self.users.values():
            if user.username == username or user.email == email:
                raise ValueError("Username or email already exists")

        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            full_name=full_name,
            roles=roles or [UserRole.USER],
            tenant_id=tenant_id,
            created_by=created_by
        )

        self.users[user.user_id] = user

        await self._log_security_event(
            event_type="user_created",
            severity="medium",
            user_id=created_by,
            details={"created_user_id": user.user_id, "username": username}
        )

        logger.info(f"Created user: {username} ({user.user_id})")
        return user

    async def authenticate_user(self,
                              username: str,
                              password: str,
                              ip_address: Optional[str] = None,
                              user_agent: Optional[str] = None) -> Optional[User]:
        """Authenticate user with username/password"""

        user = None
        for u in self.users.values():
            if u.username == username or u.email == username:
                user = u
                break

        if not user:
            await self._log_security_event(
                event_type="login_failed",
                severity="medium",
                ip_address=ip_address,
                user_agent=user_agent,
                details={"username": username, "reason": "user_not_found"}
            )
            return None

        if not user.is_account_valid():
            await self._log_security_event(
                event_type="login_blocked",
                severity="high",
                user_id=user.user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                details={"reason": "account_locked_or_expired"}
            )
            return None

        # Verify password
        if not user.password_hash or not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            user.increment_failed_login(self.max_failed_logins)

            await self._log_security_event(
                event_type="login_failed",
                severity="medium",
                user_id=user.user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                details={"reason": "invalid_password", "failed_attempts": user.failed_login_attempts}
            )
            return None

        # Successful authentication
        user.update_last_login()

        await self._log_security_event(
            event_type="login_success",
            severity="low",
            user_id=user.user_id,
            ip_address=ip_address,
            user_agent=user_agent
        )

        return user

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)

    async def update_user(self, user_id: str, updates: Dict[str, Any], updated_by: Optional[str] = None) -> bool:
        """Update user information"""
        user = await self.get_user(user_id)
        if not user:
            return False

        # Track what was changed
        changes = {}
        for field, value in updates.items():
            if hasattr(user, field) and getattr(user, field) != value:
                changes[field] = {"old": getattr(user, field), "new": value}
                setattr(user, field, value)

        if changes:
            user.updated_at = datetime.utcnow()

            await self._log_security_event(
                event_type="user_updated",
                severity="medium",
                user_id=updated_by,
                details={"updated_user_id": user_id, "changes": changes}
            )

        return True

    async def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """Change user password"""
        user = await self.get_user(user_id)
        if not user or not user.password_hash:
            return False

        # Verify old password
        if not bcrypt.checkpw(old_password.encode('utf-8'), user.password_hash.encode('utf-8')):
            await self._log_security_event(
                event_type="password_change_failed",
                severity="medium",
                user_id=user_id,
                details={"reason": "invalid_old_password"}
            )
            return False

        # Set new password
        user.password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        user.last_password_change = datetime.utcnow()
        user.password_expires_at = None  # Reset expiry

        await self._log_security_event(
            event_type="password_changed",
            severity="medium",
            user_id=user_id
        )

        return True

    # JWT Token Management
    async def create_jwt_token(self, user: User) -> JWTToken:
        """Create JWT token for user"""
        expires_at = datetime.utcnow() + timedelta(hours=self.jwt_expiry_hours)

        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "tenant_id": user.tenant_id,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user._get_role_permissions()],
            "exp": expires_at.timestamp(),
            "iat": datetime.utcnow().timestamp(),
            "iss": "multi-agent-platform"
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.jwt_algorithm)

        return JWTToken(
            token=token,
            expires_at=expires_at,
            user_id=user.user_id,
            tenant_id=user.tenant_id,
            permissions=[perm.value for perm in user._get_role_permissions()]
        )

    async def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.jwt_algorithm])

            # Check if user still exists and is valid
            user = await self.get_user(payload["user_id"])
            if not user or not user.is_account_valid():
                return None

            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None

    # API Key Management
    async def create_api_key(self,
                           name: str,
                           permissions: List[Permission],
                           user_id: Optional[str] = None,
                           tenant_id: Optional[str] = None,
                           expires_in_days: Optional[int] = None) -> Tuple[str, APIKey]:
        """Create API key and return the key value and APIKey object"""

        # Generate random API key
        key_value = f"mk_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key_value.encode()).hexdigest()

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        api_key = APIKey(
            name=name,
            key_hash=key_hash,
            permissions=permissions,
            user_id=user_id,
            tenant_id=tenant_id,
            expires_at=expires_at
        )

        self.api_keys[api_key.key_id] = api_key

        await self._log_security_event(
            event_type="api_key_created",
            severity="medium",
            user_id=user_id,
            tenant_id=tenant_id,
            details={"api_key_id": api_key.key_id, "name": name}
        )

        return key_value, api_key

    async def verify_api_key(self, key_value: str) -> Optional[APIKey]:
        """Verify API key and return APIKey object"""
        key_hash = hashlib.sha256(key_value.encode()).hexdigest()

        for api_key in self.api_keys.values():
            if api_key.key_hash == key_hash and api_key.is_valid():
                api_key.last_used = datetime.utcnow()
                return api_key

        return None

    async def revoke_api_key(self, key_id: str, revoked_by: Optional[str] = None) -> bool:
        """Revoke API key"""
        api_key = self.api_keys.get(key_id)
        if not api_key:
            return False

        api_key.is_active = False

        await self._log_security_event(
            event_type="api_key_revoked",
            severity="medium",
            user_id=revoked_by,
            details={"api_key_id": key_id, "name": api_key.name}
        )

        return True

    # Permission Checking
    async def check_permission(self,
                             user_id: Optional[str] = None,
                             api_key_id: Optional[str] = None,
                             permission: Permission = None,
                             tenant_id: Optional[str] = None) -> bool:
        """Check if user or API key has permission"""

        if user_id:
            user = await self.get_user(user_id)
            if not user or not user.is_account_valid():
                return False

            # Check tenant isolation
            if tenant_id and user.tenant_id != tenant_id and UserRole.SUPER_ADMIN not in user.roles:
                return False

            return user.has_permission(permission)

        elif api_key_id:
            api_key = self.api_keys.get(api_key_id)
            if not api_key or not api_key.is_valid():
                return False

            # Check tenant isolation
            if tenant_id and api_key.tenant_id != tenant_id:
                return False

            return permission in api_key.permissions

        return False

    # Audit and Logging
    async def _log_security_event(self,
                                event_type: str,
                                severity: str,
                                user_id: Optional[str] = None,
                                tenant_id: Optional[str] = None,
                                ip_address: Optional[str] = None,
                                user_agent: Optional[str] = None,
                                resource: Optional[str] = None,
                                action: Optional[str] = None,
                                result: str = "success",
                                details: Optional[Dict[str, Any]] = None) -> None:
        """Log security event"""

        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            tenant_id=tenant_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            result=result,
            details=details or {}
        )

        self.security_events.append(event)

        # Log to standard logger as well
        logger.info(f"Security event: {event_type} - {severity} - {result}")

        # Keep only recent events (last 10000)
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-10000:]

    async def get_security_events(self,
                                user_id: Optional[str] = None,
                                tenant_id: Optional[str] = None,
                                event_type: Optional[str] = None,
                                severity: Optional[str] = None,
                                hours: int = 24,
                                limit: int = 100) -> List[SecurityEvent]:
        """Get security events with filters"""

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        events = [e for e in self.security_events if e.timestamp >= cutoff]

        # Apply filters
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if tenant_id:
            events = [e for e in events if e.tenant_id == tenant_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if severity:
            events = [e for e in events if e.severity == severity]

        # Sort by timestamp (newest first) and limit
        events.sort(key=lambda x: x.timestamp, reverse=True)
        return events[:limit]

    # System Administration
    def get_system_security_stats(self) -> Dict[str, Any]:
        """Get system security statistics"""
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)

        recent_events = [e for e in self.security_events if e.timestamp >= last_24h]

        stats = {
            "total_users": len(self.users),
            "active_users": len([u for u in self.users.values() if u.last_login and u.last_login >= last_24h]),
            "locked_accounts": len([u for u in self.users.values() if u.account_locked]),
            "total_api_keys": len(self.api_keys),
            "active_api_keys": len([k for k in self.api_keys.values() if k.is_active]),
            "security_events_24h": len(recent_events),
            "failed_logins_24h": len([e for e in recent_events if e.event_type == "login_failed"]),
            "successful_logins_24h": len([e for e in recent_events if e.event_type == "login_success"]),
            "high_severity_events_24h": len([e for e in recent_events if e.severity == "high"]),
            "critical_events_24h": len([e for e in recent_events if e.severity == "critical"])
        }

        return stats

    async def cleanup_expired_tokens_and_keys(self) -> None:
        """Clean up expired API keys and other expired data"""
        now = datetime.utcnow()

        # Mark expired API keys as inactive
        expired_keys = 0
        for api_key in self.api_keys.values():
            if api_key.expires_at and now > api_key.expires_at and api_key.is_active:
                api_key.is_active = False
                expired_keys += 1

        if expired_keys > 0:
            logger.info(f"Marked {expired_keys} API keys as expired")

        # Clean up old security events (keep last 30 days)
        cutoff = now - timedelta(days=30)
        original_count = len(self.security_events)
        self.security_events = [e for e in self.security_events if e.timestamp >= cutoff]

        if len(self.security_events) < original_count:
            logger.info(f"Cleaned up {original_count - len(self.security_events)} old security events")