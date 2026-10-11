SELECT
    MIN(subject)                AS example_subject,
    COUNT(*)                    AS messages,
    MIN(sent_at)::DATE          AS first_seen,
    MAX(sent_at)::DATE          AS last_seen
FROM messages
WHERE label = 1
GROUP BY template_key
HAVING COUNT(*) >= 3
ORDER BY messages DESC, first_seen, template_key
LIMIT 25;
