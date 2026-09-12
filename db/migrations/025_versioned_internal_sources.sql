-- Bulletins, SOPs, and the formulary are issued in versions (a bulletin
-- supersedes an earlier one on the same subject; an SOP has archived prior
-- versions; the formulary is per plan year). Their manifests now carry
-- effective_from / effective_to, so the version-precedence filter that
-- clinical policies use applies to them too: a default question sees the
-- version in effect, a dated question the version in effect on that date.
UPDATE sources SET versioned = true WHERE key IN ('bulletins', 'sops', 'formulary');
