# Demo script — 12 minutes

Beat by beat. Times are cumulative. The italics are things worth actually
saying out loud; everything else is stage direction.

---

### 0:00 — Frame it before you touch the keyboard (1 min)

Don't open with the technology. Open with the constraint.

*"Telescope time is the scarcest thing in my field. You apply for it, you
usually don't get it, and when you do, the atmosphere is still in the way.
Every image any telescope has ever taken is blurrier than the sky actually
is. What I'm going to show you is one of the ways we're now clawing some of
that back — and, more to the point, what a student would learn by building
it."*

Then say the honest thing up front, because it buys you credibility for the
rest:

*"This model trained in about ten minutes yesterday. It is not
sophisticated. A motivated sophomore could build it."*

---

### 1:00 — Cell 1: load the model (30 sec)

It prints the number of parameters and the training time. Read it out.

*"Half a million numbers. That's small — the models you've read about in the
news have a million times more. This one runs on a laptop."*

---

### 1:30 — Cell 2: the truth image (1 min)

*"This is a real galaxy, from a public archive that anyone in this room can
download from tonight. Remember what it looks like — it's the answer key."*

---

### 2:30 — Cell 3: ruin it on purpose (1.5 min)

This is the beat most people find counterintuitive, so slow down here.

*"To teach the computer to fix blurry images, I need pairs — a bad image and
the correct answer. Those don't exist in nature. So I take a good image and
damage it on purpose: blur it like the atmosphere does, throw away fifteen
of every sixteen pixels, add the noise a real detector adds. Now I have a
matched pair, and I can grade the machine."*

Then ask the room, and wait for an answer:

*"How much of that do you think is recoverable?"*

Most rooms say "almost none." Let that sit.

---

### 4:00 — Cell 4: the reveal (2 min)

Run it. Don't narrate while it renders. Let them look.

Point at panel two first: *"That's the ordinary way to enlarge an image —
what your phone does when you pinch to zoom. No AI."* Then panel three:
*"That's the network."* Then panel four: *"And that's the truth, for
comparison."*

*"It took a fifth of a second."*

---

### 6:00 — Cell 5: zoom in (1 min)

*"Same square of sky, blown up. This is where you can see what it actually
recovered — and where it didn't."*

---

### 7:00 — Cells 6–7: where it's wrong (2 min) ← **the important one**

This is the beat that wins the argument. Don't rush it.

*"Here's the part I'd want a student to understand before anything else.
This model doesn't reveal hidden detail. It has learned what galaxies tend
to look like, and it uses that expectation to guess. Usually the guess is
good. Sometimes it invents a feature that isn't there — a perfectly
plausible smudge where the real sky had nothing."*

*"That's why nobody publishes an AI-sharpened image as evidence of a
discovery. We use it to decide where to point the expensive telescope next.
The AI narrows the search. A human confirms the finding."*

If you only get one sentence into their heads today, make it that one.

---

### 9:00 — Cell 8: the scoreboard (1 min)

*"And this is measured on galaxies the model never saw — because otherwise
I'd just be showing you that it memorised its homework."*

---

### 10:00 — Cells 9–10: let them pick one (1 min)

*"Someone call one out."*

Type the name, run it. This is the moment the room stops watching a
presentation and starts watching a tool. Worth the minute.

---

### 11:00 — Cell 11: the arithmetic (1 min)

*"One galaxy is a party trick. Here's why observatories actually care."*

Read the years-of-human-labour number out loud. *"That's the reason the field
adopted this. Not novelty. Arithmetic."*

---

### 12:00 — Close on the table, not the galaxy (1–2 min)

Scroll to the final table. Read the left column, then the right.

*"Nothing in that left column is astronomy. Everything in the right column
is on a job posting somewhere right now."*

Then the ask:

*"The two rows I care about are the last two. A student who has watched a
model produce a confident, beautiful, wrong answer — in a case where they
happen to be holding the answer key — has learned something about AI that I
cannot teach them by warning them about it. They're going to use these tools
either way. The question is whether they learn to check them here, with a
galaxy, where being wrong costs nothing — or later, somewhere it doesn't."*

Stop there. Don't add a summary.

---

## Questions you will probably get

**"Isn't it just making things up?"**
Partly, yes — and that's the honest answer. It's making a statistically
informed guess. It's right on average, which is enough to prioritise where
to look, and not enough to claim a discovery. That distinction is the whole
skill.

**"Could a student really build this?"**
The code is about 700 lines, and half of that is comments. A student with one
semester of Python could build a rougher version in a weekend. The hard part
isn't the network — it's designing the test that tells you whether it worked.

**"What about cheating?"**
Worth conceding directly: the thing that makes this useful in a classroom is
that the student has to produce an *evaluation*, not an answer. You can't
fake the held-out score. Assignments that ask "did it work, and how do you
know?" are much harder to cheat than assignments that ask for an answer.

**"Do we need GPUs?"**
Not for this. It trains on a laptop overnight or a free Colab GPU in ten
minutes. Cost of entry is zero.

**"What would this look like in my subject?"**
Same skeleton, different data: damaged historical text, noisy audio, low-res
microscope images, sensor data with dropouts. The pattern — make pairs, train,
hold data back, check where it fails — transfers to every one of them.
