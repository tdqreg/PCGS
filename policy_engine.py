from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ----------------------------------------------------------------------
# Authorization State
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class AuthState:
    """
    Path-dependent authorization state.

    grants:
        Permissions accumulated along the current path.

    clearance:
        Current effective clearance.

    denied:
        Whether the current path has entered an explicit denied state.
    """

    grants: frozenset[str] = field(
        default_factory=frozenset
    )

    clearance: int = 0

    denied: bool = False

    def has(
        self,
        permission: str,
    ) -> bool:

        return permission in self.grants

    def with_grant(
        self,
        permission: str,
    ) -> "AuthState":

        return AuthState(
            grants=(
                self.grants
                | {permission}
            ),
            clearance=self.clearance,
            denied=self.denied,
        )

    def with_clearance(
        self,
        clearance: int,
    ) -> "AuthState":

        return AuthState(
            grants=self.grants,
            clearance=max(
                self.clearance,
                clearance,
            ),
            denied=self.denied,
        )


# ----------------------------------------------------------------------
# Transition Result
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class TransitionResult:
    allowed: bool
    next_state: AuthState
    reason: str


# ----------------------------------------------------------------------
# Policy Engine
# ----------------------------------------------------------------------

class PolicyEngine:
    """
    Deterministic policy engine for PCGS Level-2.

    Authorization rules are stored directly on graph edges.

    Edge schema:

        {
            "edge_id": "...",
            "source": "...",
            "target": "...",
            "cost": 1.0,
            "required_grants": ["manager"],
            "min_clearance": 2,
            "grant": "viewer",
            "grant_clearance": 3
        }

    Authorization semantics:

        1. The current authorization state must satisfy all
           required grants.

        2. The current authorization state must satisfy the
           minimum clearance.

        3. If authorized, the edge may propagate a new grant.

        4. If authorized, the edge may propagate a higher
           clearance.

    This implements explicit propagation grant semantics.
    """

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    def initial_state(
        self,
        graph: Any,
    ) -> AuthState:

        # The Level-2 generator stores policy configuration
        # under metadata["policy"].

        policy = graph.metadata.get(
            "policy",
            {},
        )

        grants = frozenset(
            policy.get(
                "initial_grants",
                [],
            )
        )

        clearance = int(
            policy.get(
                "initial_clearance",
                0,
            )
        )

        return AuthState(
            grants=grants,
            clearance=clearance,
            denied=False,
        )

    # ------------------------------------------------------------------
    # Edge transition
    # ------------------------------------------------------------------

    def transition(
        self,
        graph: Any,
        state: AuthState,
        edge: Any,
    ) -> TransitionResult:

        # --------------------------------------------------------------
        # Already denied
        # --------------------------------------------------------------

        if state.denied:

            return TransitionResult(
                allowed=False,
                next_state=state,
                reason="already_denied",
            )

        # --------------------------------------------------------------
        # Required grants
        #
        # IMPORTANT:
        # Read authorization directly from the Edge object.
        # --------------------------------------------------------------

        required_grants = set(
            getattr(
                edge,
                "required_grants",
                [],
            )
        )

        missing = (
            required_grants
            - set(state.grants)
        )

        if missing:

            return TransitionResult(
                allowed=False,
                next_state=state,
                reason=(
                    "missing_grants:"
                    + ",".join(
                        sorted(missing)
                    )
                ),
            )

        # --------------------------------------------------------------
        # Minimum clearance
        # --------------------------------------------------------------

        required_clearance = int(
            getattr(
                edge,
                "min_clearance",
                0,
            )
        )

        if (
            state.clearance
            < required_clearance
        ):

            return TransitionResult(
                allowed=False,
                next_state=state,
                reason=(
                    "insufficient_clearance:"
                    f"{state.clearance}"
                    f"<{required_clearance}"
                ),
            )

        # --------------------------------------------------------------
        # Explicit deny
        #
        # The current Level-2 dataset does not require this field,
        # but keeping the check makes the policy engine extensible.
        # --------------------------------------------------------------

        if getattr(
            edge,
            "deny",
            False,
        ):

            denied_state = AuthState(
                grants=state.grants,
                clearance=state.clearance,
                denied=True,
            )

            return TransitionResult(
                allowed=False,
                next_state=denied_state,
                reason="explicit_deny",
            )

        # --------------------------------------------------------------
        # Authorized transition.
        # --------------------------------------------------------------

        next_state = state

        # --------------------------------------------------------------
        # Grant propagation
        # --------------------------------------------------------------

        propagated_grant = getattr(
            edge,
            "grant",
            None,
        )

        if propagated_grant is not None:

            next_state = (
                next_state.with_grant(
                    propagated_grant
                )
            )

        # --------------------------------------------------------------
        # Clearance propagation
        # --------------------------------------------------------------

        propagated_clearance = getattr(
            edge,
            "grant_clearance",
            None,
        )

        if propagated_clearance is not None:

            next_state = (
                next_state.with_clearance(
                    int(
                        propagated_clearance
                    )
                )
            )

        # --------------------------------------------------------------
        # Successful transition
        # --------------------------------------------------------------

        return TransitionResult(
            allowed=True,
            next_state=next_state,
            reason="allow",
        )

    # ------------------------------------------------------------------
    # Dominance
    # ------------------------------------------------------------------

    def state_dominates(
        self,
        stronger: AuthState,
        weaker: AuthState,
    ) -> bool:
        """
        Return True when `stronger` has authorization capabilities
        that are at least as permissive as `weaker`.

        Under the monotonic authorization assumption:

            grants(stronger) ⊇ grants(weaker)

        and

            clearance(stronger) >= clearance(weaker)

        imply that every future authorization transition available
        to `weaker` is also available to `stronger`.

        A denied state cannot dominate a non-denied state.
        A non-denied state dominates a denied state because the
        denied state cannot produce an authorized continuation.
        """

        if stronger.denied:

            return False

        if weaker.denied:

            return True

        return (
            stronger.grants.issuperset(
                weaker.grants
            )
            and
            stronger.clearance
            >= weaker.clearance
        )