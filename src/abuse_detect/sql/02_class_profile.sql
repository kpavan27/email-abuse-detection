-- corpus is pre-tokenised: 'http : / / www . x . com'
SELECT
    CASE label WHEN 1 THEN 'spam' ELSE 'ham' END                       AS class,
    COUNT(*)                                                           AS messages,
    quantile_cont(length(body), 0.5)                                   AS median_body_chars,
    quantile_cont(length(body), 0.9)                                   AS p90_body_chars,
    AVG(CASE WHEN regexp_matches(body, '(?i)https?\s*:\s*/\s*/|www\s*\.') THEN 1 ELSE 0 END) AS share_with_url,
    AVG(CASE WHEN regexp_matches(body, '(?i)[a-z][|~%!][a-z]|[a-z]\s[|~]\s?[a-z]|[a-z]@[a-z]') THEN 1 ELSE 0 END) AS share_symbol_obfuscation,
    AVG(CASE WHEN regexp_matches(subject, '(?i)^\s*(re|fw|fwd)\s*:') THEN 1 ELSE 0 END) AS share_reply_fwd,
    AVG(CASE WHEN regexp_matches(lower(body), '\benron\b') THEN 1 ELSE 0 END) AS share_mentions_enron
FROM messages
GROUP BY 1
ORDER BY 1;
