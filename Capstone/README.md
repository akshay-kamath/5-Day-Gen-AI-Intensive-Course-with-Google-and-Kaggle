# ChetnaShakti: The Power of Conscious Self-Therapy & Habit Rewiring

**An AI-Powered CBT and Spiritual Wellness Coach using Gemini 2.5**

---

## About the Project

Modern mental health challenges like overthinking, anxiety, burnout, and emotional disconnection are deeply tied to negative thought patterns and misaligned habits. While Cognitive Behavioral Therapy (CBT) offers evidence-based tools for change, many individuals also seek a connection to spiritual wisdom that resonates with their inner self.

**ChetnaShakti** bridges psychology and spirituality by uniting:
- CBT-based journaling and reflection
- Habit rewiring frameworks using behavior loops and micro-goals
- Scriptural wisdom from both the **Bhagavad Gita** and **Durga Saptashati**
- The power of advanced AI models (Gemini 2.5) and multimodal GenAI features

---

## Key Objectives
- Help users track and transform emotional cycles
- Enable habit transformation through micro-interventions
- Offer scripturally grounded spiritual support tailored to emotional states
- Provide daily rituals, affirmations, journaling prompts, and action plans

---

## GenAI Features Used in ChetnaShakti

### Structured Output / JSON Mode
Used for generating CBT action plans, journaling templates, and daily rituals in a machine-readable format for UI export.

**Example Output:**
```json
{
  "daily_goal": "Reduce anxious overthinking",
  "action_steps": [
    "Practice 5-minute breathing before work",
    "Challenge negative thoughts with evidence",
    "Reflect on 1 thing I did well today"
  ],
  "mantra": "I am steady, strong, and supported by the universe."
}
```

### Few-Shot Prompting
Provides spiritual and psychological few-shot examples using both Bhagavad Gita and Durga Saptashati to train the model for structured, insightful generation.

### Function Calling
Triggers internal agents:
- `CBTAgent`: Detects cognitive distortions, offers therapeutic techniques
- `HabitAgent`: Builds atomic habits and tracks rituals
- `SpiritualAgent`: Retrieves mantras and verses from scriptures

### Document Understanding
Processes long-form spiritual or psychological texts like full chapters of Gita or Durga Saptashati and extracts relevant segments.

### Image Understanding
Analyzes vision boards, journal photos, or sacred scans (e.g., Devi yantras, Gita verses) for emotional and symbolic content.

**Use Case:**
Image of mood journal with negative keywords is interpreted for emotional tone and spiritual recommendation.

### Video Understanding
Analyzes user-uploaded videos for emotional states using tone, expression, and gesture.

**Use Case:**
User shares a video expressing burnout. The AI recommends a Gita verse on detachment and a Durga mantra on energetic renewal.

### Audio Understanding
Transcribes and analyzes voice notes for emotion and intent.

**Use Case:**
User says, "I can’t stop overthinking." The system recommends a guided Durga mantra chant and reframing journaling prompt.

### Long Context Window
Tracks emotional, behavioral, and spiritual evolution over long periods using past journaling or scripture engagement.

**Use Case:**
After 30+ days of entries, system suggests deeper mantra programs and relevant verses.

### Context Caching
Personalizes future interactions by remembering prior mantras, verses, emotional states, and resonance history.

**Use Case:**
Avoids repeating already-used insights unless therapeutically useful.

### Grounding
All insights are based on:
- Verified CBT models
- Cited scriptural sources from Gita and Durga Saptashati
- Avoids hallucinations by ensuring authenticity

### Vector Search / Vector Store / Vector Database
Embeds user logs, moods, spiritual goals, and verses into a vector database to enable semantic matching.

**Use Case:**
User writes: "I feel lost and hopeless."
System returns:
- Gita 2.3: "Yield not to unmanliness..."
- Durga Saptashati verse on Devi as the destroyer of fear
- Matching micro-habit and journaling reflection

---

## Example Interaction

**User Journal:**
"I feel stuck. I keep procrastinating even though I want to be productive."

**ChetnaShakti Responds:**
```json
{
  "bhagavad_gita_insight": {
    "verse": "Chapter 2, Verse 47",
    "sanskrit": "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन",
    "translation": "You have a right to perform your duty, but not to the fruits of your actions.",
    "interpretation": "Focus on action, not outcomes. Attachment to results creates paralysis. Just begin.",
    "mantra": "Karmanye Vadhikaraste"
  },
  "durga_saptashati_insight": {
    "verse": "Chapter 8, Verse 9",
    "sanskrit": "सर्वस्वरूपे सर्वेशे सर्वशक्तिसमन्विते ।\nभयेभ्यस्त्राहि नो देवी दुर्गे देवि नमोऽस्तुते ॥",
    "translation": "O Durga, who exists in all forms, protect us from fear. Salutations to you.",
    "interpretation": "Invoke the Devi when fear and inaction paralyze you. Her power awakens inner movement."
  },
  "cbt_plan": {
    "daily_goal": "Overcome task avoidance",
    "action_steps": [
      "Break large task into 3 small chunks",
      "Do a 5-minute countdown timer to start",
      "Celebrate small wins with gratitude journal"
    ],
    "affirmation": "I move forward with ease and trust in the process."
  }
}
```

---

## Future Enhancements
- Integration with voice journaling apps
- Mobile UI with drag-and-drop sacred rituals
- Personalized mantra playlists based on emotional graph

---

## License & Credits
- Uses Gemini 2.5 (Gemini API)
- Verses sourced from public domain translations of the Bhagavad Gita and Durga Saptashati
- CBT models adapted from validated therapy frameworks

---



