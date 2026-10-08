SELECT
    date_trunc('quarter', sent_at) AS quarter,
    COUNT(*)                       AS spam_messages,
    ROUND(AVG(CASE WHEN regexp_matches(body, '(?i)https?\s*:\s*/\s*/|www\s*\.') THEN 1 ELSE 0 END), 3) AS share_with_url
FROM messages
WHERE label = 1 AND sent_at IS NOT NULL
GROUP BY 1
HAVING COUNT(*) >= 100
ORDER BY 1;
