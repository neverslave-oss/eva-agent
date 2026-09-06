# Driver contract

All drivers must expose:
- observe(target) -> Observation
- execute(actions, target) -> ExecutionResult
- verify(expectation, target) -> VerificationResult (optional specialization)
