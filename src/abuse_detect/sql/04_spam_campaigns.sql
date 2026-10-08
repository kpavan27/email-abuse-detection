SELECT
    any_value(subject)          AS example_subject,
    COUNT(*)                    AS messages,
    MIN(sent_at)::DATE          AS first_seen,
    MAX(sent_at)::DATE          AS last_seen
FROM messages
WHERE label = 1
GROUP BY template_key
HAVING COUNT(*) >= 3
ORDER BY messages DESC
LIMIT 25;
