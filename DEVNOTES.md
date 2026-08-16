# Developer notes — not shipped to users

This repo publishes a **plugin** (the load-bearing artifact). The companion **skill**
(`skills/deep-think/SKILL.md`) is authored locally on the origin machine via
`skill_manage` because user-local skills live outside any single repo. If we later
contribute the skill upstream to hermes-agent, it moves to `skills/` here and follows
the in-repo authoring standard (tests + docs regen) at that point.
