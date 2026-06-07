# markov-blanket

> **Where does an agent end and the world begin? The Markov blanket knows.**

[![crates.io](https://img.shields.io/crates/v/markov-blanket.svg)](https://crates.io/crates/markov-blanket)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Implements the Markov blanket — the set of states that shield an agent from the rest of the universe. In the Free Energy Principle, the blanket separates internal states from external states, and everything an agent can know must come through it.

## The Free Energy Principle

Karl Friston's Free Energy Principle states that agents minimize variational free energy — a bound on surprise. The **Markov blanket** is the statistical boundary:
- **Internal states**: the agent's own model
- **External states**: the environment
- **Sensory states**: what the agent observes (blanket)
- **Active states**: what the agent does (blanket)

The agent can never directly access external states — only through the blanket.

## License

MIT © [SuperInstance](https://github.com/SuperInstance)

Part of the [Exocortex](https://github.com/SuperInstance/exocortex) project.
