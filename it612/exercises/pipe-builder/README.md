# In-Class Exercise: Pipe Builder

In this exercise you'll build a pipeline one piece at a time, then turn it into a reusable script. Each part builds on the last.

## Setup

### First time? Fork the repo

1. Go to [github.com/blkfin/courses](https://github.com/blkfin/courses) and click the **Fork** button (top right). This creates a copy under your own GitHub account.
2. From **your fork** (it should say `yourusername/courses` at the top), click the green **Code** button → **Codespaces** tab → **Create codespace on main**.
3. You now have a full Linux terminal in your browser. Verify it works:
   ```bash
   ls it612/exercises/pipe-builder/
   ```
   You should see this README and the sample files.

### Already forked from a previous assignment?

Open your existing Codespace and pull the latest updates:
```bash
git remote add upstream https://github.com/blkfin/courses.git   # only needed once
git fetch upstream
git merge upstream/main
```

### Navigate to the exercise

```bash
cd it612/exercises/pipe-builder/
```

Take a quick look at what's here:
```bash
ls
```

You should see `README.md`, `sample-passwd.txt`, and `duplicates-test.txt`.

---

## Background: What is `/etc/passwd`?

Every Linux system has a file called `/etc/passwd` that lists all user accounts. Each line is one user, with fields separated by colons (`:`):

```
username:password:uid:gid:description:home_directory:shell
```

For example:
```
alice:x:1001:1001:Alice Chen:/home/alice:/bin/bash
```

- `alice` — the username
- `x` — password (stored elsewhere for security)
- `1001` — user ID number
- `1001` — group ID number
- `Alice Chen` — description / full name
- `/home/alice` — home directory
- `/bin/bash` — default shell

We've included a `sample-passwd.txt` file so everyone is working with the same data. It has 34 users — a mix of system accounts (like `root`, `www-data`, `nginx`) and regular users (like `alice`, `bob`, `charlie`).

---

## Part 1 — Build a Pipeline (5 min)

The goal: **count how many users are in the file.** But instead of writing one big command, build it up step by step. Run each line and look at the output before moving on.

**Step 1 — See the whole file:**
```bash
cat sample-passwd.txt
```
You should see all 34 lines. That's a lot of output — let's narrow it down.

**Step 2 — Extract just the usernames:**

The `cut` command pulls out specific fields from structured text. Since each field is separated by `:`, we tell `cut` to use that as the delimiter and grab field 1 (the username):

```bash
cat sample-passwd.txt | cut -d':' -f1
```

Now you should see just the usernames, one per line. Much cleaner.

**Step 3 — Sort them:**
```bash
cat sample-passwd.txt | cut -d':' -f1 | sort
```

Same usernames, now in alphabetical order.

**Step 4 — Count them:**
```bash
cat sample-passwd.txt | cut -d':' -f1 | sort | wc -l
```

`wc -l` counts lines. You should get `34`.

> **What just happened?** You built a pipeline — each command's output feeds into the next command's input via the `|` (pipe) character. Data flows left to right: `cat` → `cut` → `sort` → `wc`.

---

## Part 2 — Save It as a Script (5 min)

Typing that pipeline every time is tedious. Let's make it a reusable script.

**Step 1 — Create the script file:**

Open a new file in the editor (click **File → New File** in the Codespace menu bar, or use the terminal):
```bash
touch usercount.sh
```

Then open it in the editor by clicking on it in the file explorer on the left.

**Step 2 — Write the script:**

Add these lines to `usercount.sh`:
```bash
#!/bin/bash
set -e
cat sample-passwd.txt | cut -d':' -f1 | sort | wc -l
```

- `#!/bin/bash` — tells the system this is a bash script (called a "shebang" line)
- `set -e` — stop immediately if any command fails (good practice)

Save the file (**Ctrl+S**).

**Step 3 — Run it:**
```bash
bash usercount.sh
```

You should get `34` — same result as before, but now it's saved.

**Step 4 — Make it flexible:**

Right now the filename is hardcoded. What if you want to analyze a different file? Replace `sample-passwd.txt` with `"$1"` — this is a special variable that holds the first argument you pass to the script:

```bash
#!/bin/bash
set -e
cat "$1" | cut -d':' -f1 | sort | wc -l
```

> **Why the quotes around `$1`?** If the filename has spaces in it (like `"my file.txt"`), quotes keep it treated as one argument instead of two.

Save the file and test it:
```bash
bash usercount.sh sample-passwd.txt
```

Still `34`? Great — same result, but now it works on any file you give it.

---

## Part 3 — Make It Useful (3 min)

**Step 1 — Try it on the real system file:**
```bash
bash usercount.sh /etc/passwd
```

It works on any file with the same format — that's the power of making it flexible with `$1`.

**Step 2 — Test with duplicates:**

We've included a test file that has duplicate usernames. Take a look:
```bash
cat duplicates-test.txt
```

Now run your script on it:
```bash
bash usercount.sh duplicates-test.txt
```

You should get `8` — but there are only 5 unique usernames. The script is counting duplicates.

**Step 3 — Fix it:**

Add `uniq` to your pipeline (after `sort`, before `wc`). The `uniq` command removes **adjacent** duplicate lines — that's why we sort first, so duplicates are next to each other.

Update `usercount.sh`:
```bash
#!/bin/bash
set -e
cat "$1" | cut -d':' -f1 | sort | uniq | wc -l
```

Test it again:
```bash
bash usercount.sh duplicates-test.txt
```

Now you should get `5` — just the unique usernames.

Verify it still works on the full file:
```bash
bash usercount.sh sample-passwd.txt
```

Should still be `34` (no duplicates in that file).

---

## Submission

```bash
git add usercount.sh
git commit -m "Pipe builder exercise complete"
git push
```

Submit the URL to your GitHub repository on Canvas.
