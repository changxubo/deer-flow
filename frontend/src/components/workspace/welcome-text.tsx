'use client';

import { memo, useEffect, useMemo, useState } from 'react';
import { useI18n } from '@/core/i18n/hooks';

// A simple shuffle implementation to replace es-toolkit/compat
const shuffle = <T,>(array: T[]): T[] => {
  const newArray = [...array];
  for (let i = newArray.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const temp = newArray[i];
    newArray[i] = newArray[j]!;
    newArray[j] = temp!;
  }
  return newArray;
};

const LoadingDots = ({ className }: { className?: string }) => (
  <div className={`flex items-center space-x-1 ${className}`}>
    <span className="h-5 w-5 ml-2 animate-pulse rounded-full bg-current [animation-delay:0s]" />
  </div>
);

const TypewriterEffect = ({
  sentences,
  typingSpeed = 64,
  deletingSpeed = 32,
  pauseDuration = 16000,
  deletePauseDuration = 1000,
  cursorCharacter = <LoadingDots />,
  hideCursorWhileTyping,
}: {
  sentences: string[];
  typingSpeed?: number;
  deletingSpeed?: number;
  pauseDuration?: number;
  deletePauseDuration?: number;
  cursorCharacter?: React.ReactNode;
  hideCursorWhileTyping?: 'afterTyping' | 'whileTyping';
}) => {
  const [sentenceIndex, setSentenceIndex] = useState(0);
  const [text, setText] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);
  const [isTyping, setIsTyping] = useState(true);

  useEffect(() => {
    const currentSentence = sentences[sentenceIndex];
    if (!currentSentence) return;

    const handleTyping = () => {
      if (isDeleting) {
        if (text.length > 0) {
          setText((prev) => prev.substring(0, prev.length - 1));
        } else {
          setIsDeleting(false);
          setSentenceIndex((prev) => (prev + 1) % sentences.length);
          // Pause before typing next sentence
          setTimeout(() => {}, deletePauseDuration);
        }
      } else {
        if (text.length < currentSentence.length) {
          setText((prev) => currentSentence.substring(0, prev.length + 1));
        } else {
          setTimeout(() => setIsDeleting(true), pauseDuration);
        }
      }
    };

    const timeout = setTimeout(handleTyping, isDeleting ? deletingSpeed : typingSpeed);
    setIsTyping(!isDeleting && text.length < currentSentence.length);

    return () => clearTimeout(timeout);
  }, [
    text,
    isDeleting,
    sentenceIndex,
    sentences,
    typingSpeed,
    deletingSpeed,
    pauseDuration,
    deletePauseDuration,
  ]);

  const showCursor = !(hideCursorWhileTyping === 'whileTyping' && isTyping) &&
    !(hideCursorWhileTyping === 'afterTyping' && !isTyping) &&
    !(!isTyping && !isDeleting);


  return (
    <>
      {text}
      {showCursor && cursorCharacter}
    </>
  );
};

const WelcomeText = memo(() => {
  const { t } = useI18n();
  const locale = t.locale.localName;
  const sentences = useMemo(() => {
    if (!t.welcome?.welcomeMessages) return [];
    const messages = t.welcome.welcomeMessages;
    return shuffle(messages);
  }, [t]);

  if (sentences.length === 0) {
    return null;
  }

  return (
    <div
      className="flex items-center justify-center"
      style={{
        fontSize: 28,
        fontWeight: 'bold',
        marginBlock: '36px 24px',
      }}
    >
      <TypewriterEffect
        cursorCharacter={<LoadingDots className="text-2xl" />}
        key={locale}
        sentences={sentences}
      />
    </div>
  );
});

export default WelcomeText;
