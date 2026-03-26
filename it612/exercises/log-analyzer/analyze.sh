#!/bin/bash
# Log Analyzer — IT 612 Exercise
# Analyze server.log using command-line tools
#
# Run:  bash analyze.sh
# Save: bash analyze.sh | tee report.txt

LOG="server.log"

echo "=== Log Analysis Report ==="
echo ""

# ─────────────────────────────────────────────
# Step 1: Count the Basics
# Use grep and wc to count lines by log level.
# ─────────────────────────────────────────────
echo "--- Line Counts ---"

echo "Total lines: $(wc -l < server.log)"
echo "Error lines: $(grep "ERROR" server.log | wc -l)"
echo "Warning lines: $(grep "WARN" server.log | wc -l)"

echo ""

# ─────────────────────────────────────────────
# Step 2: Extract Unique Errors
# Find all ERROR lines, extract the message,
# then remove duplicates.
# ─────────────────────────────────────────────
echo "--- Unique Error Messages ---"

grep "ERROR" server.log | awk '{for(i=4;i<=NF;i++) printf "%s ", $i; print ""}' | sort | uniq

echo ""

# ─────────────────────────────────────────────
# Step 3: Most Requested Endpoints
# Find GET/POST requests, count unique
# method+path combinations, rank by frequency.
# ─────────────────────────────────────────────
echo "--- Top Endpoints ---"

grep -E "GET|POST" server.log | awk '{print $5, $6}' | sort | uniq -c | sort -rn

echo ""

# ─────────────────────────────────────────────
# Step 4: Who Logged In?
# Find login sessions and count per user.
# ─────────────────────────────────────────────
echo "--- User Logins ---"

grep "session created for user=" server.log | grep -o 'user=[a-z]*' | sort | uniq -c | sort -rn

echo ""

# ─────────────────────────────────────────────
# Step 5: Save the Report
# Add a timestamp line so you know when this ran.
# ─────────────────────────────────────────────

echo "Report generated: $(date)"
echo


# ─────────────────────────────────────────────
# Extra: Error Rate
# Calculates the percentage of logs that are errors
# ─────────────────────────────────────────────
echo "--- Error Rate ---"

total=$(wc -l < server.log)
errors=$(grep "ERROR" server.log | wc -l)

echo "Error rate: $((errors * 100 / total))%"

echo ""
