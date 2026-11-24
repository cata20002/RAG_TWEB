import { Component, inject, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

// Angular Material Imports
import { MatCardModule } from '@angular/material/card';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatTooltipModule } from '@angular/material/tooltip';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: { intent: string; content: string }[];
  question?: string;
  feedback?: 'thumbs_up' | 'thumbs_down' | null;
  feedbackSubmitting?: boolean;
  feedbackComment?: string;
  showCommentBox?: boolean;
  feedbackSent?: boolean;
}

@Component({
  selector: 'app-rag-chat',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatExpansionModule,
    MatSnackBarModule,
    MatTooltipModule
  ],
  template: `
    <div class="chat-container">
      <mat-card class="chat-card mat-elevation-z4">
        <mat-card-header>
          <div mat-card-avatar><mat-icon>support_agent</mat-icon></div>
          <mat-card-title>Customer Support AI</mat-card-title>
          <mat-card-subtitle>Powered by local RAG Pipeline</mat-card-subtitle>
        </mat-card-header>

        <mat-card-content class="chat-messages">
          <!-- Chat History -->
          <div *ngFor="let msg of messages; let i = index; trackBy: trackByIndex" class="message-row" [ngClass]="msg.role">
            <div class="message-bubble">
              <p>{{ msg.content }}</p>
              
              <!-- Collapsible Sources (Only shows for AI messages with context) -->
              <mat-accordion *ngIf="msg.sources && msg.sources.length > 0">
                <mat-expansion-panel class="mat-elevation-z1">
                  <mat-expansion-panel-header>
                    <mat-panel-title class="source-title">
                      <mat-icon class="source-icon">library_books</mat-icon> View Sources
                    </mat-panel-title>
                  </mat-expansion-panel-header>
                  
                  <div *ngFor="let src of msg.sources; let j = index" class="source-item">
                    <strong>Source {{ j + 1 }} (Intent: {{ src.intent || 'N/A' }}):</strong>
                    <p>{{ src.content }}</p>
                  </div>
                </mat-expansion-panel>
              </mat-accordion>

              <!-- Feedback Section (Only for assistant messages) -->
              <div *ngIf="msg.role === 'assistant'" class="feedback-section">
                <div class="feedback-actions">
                  <button
                    mat-icon-button
                    [color]="msg.feedback === 'thumbs_up' ? 'primary' : undefined"
                    matTooltip="Helpful"
                    [disabled]="msg.feedbackSubmitting || msg.feedbackSent"
                    (click)="toggleRating(i, 'thumbs_up')"
                    class="feedback-btn"
                  >
                    <mat-icon>{{ msg.feedback === 'thumbs_up' ? 'thumb_up' : 'thumb_up_off_alt' }}</mat-icon>
                  </button>
                  <button
                    mat-icon-button
                    [color]="msg.feedback === 'thumbs_down' ? 'warn' : undefined"
                    matTooltip="Not helpful"
                    [disabled]="msg.feedbackSubmitting || msg.feedbackSent"
                    (click)="toggleRating(i, 'thumbs_down')"
                    class="feedback-btn"
                  >
                    <mat-icon>{{ msg.feedback === 'thumbs_down' ? 'thumb_down' : 'thumb_down_off_alt' }}</mat-icon>
                  </button>
                  <span *ngIf="msg.feedbackSent" class="feedback-thanks">Thanks for your feedback!</span>
                </div>

                <!-- Comment box appears after selecting a rating -->
                <div *ngIf="msg.feedback && !msg.feedbackSent" class="feedback-comment-area">
                  <mat-form-field appearance="outline" class="feedback-comment-field">
                    <mat-label>Add a comment (optional)</mat-label>
                    <textarea
                      matInput
                      [value]="msg.feedbackComment || ''"
                      (input)="updateComment(i, $event)"
                      rows="2"
                      placeholder="Tell us more about your experience..."
                    ></textarea>
                  </mat-form-field>
                  <button
                    mat-flat-button
                    color="primary"
                    [disabled]="msg.feedbackSubmitting"
                    (click)="submitFeedback(i)"
                    class="submit-feedback-btn"
                  >
                    <mat-spinner *ngIf="msg.feedbackSubmitting" diameter="18" class="btn-spinner"></mat-spinner>
                    <span *ngIf="!msg.feedbackSubmitting">Submit Feedback</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
          
          <!-- Loading Spinner -->
          <div *ngIf="isLoading" class="loading-indicator">
            <mat-spinner diameter="30"></mat-spinner>
          </div>
        </mat-card-content>

        <!-- Input Area -->
        <mat-card-actions class="chat-input-area">
          <mat-form-field appearance="outline" class="full-width">
            <mat-label>Ask a question...</mat-label>
            <input matInput [(ngModel)]="userInput" (keyup.enter)="sendMessage()" [disabled]="isLoading">
            <button matSuffix mat-icon-button color="primary" (click)="sendMessage()" [disabled]="isLoading || !userInput.trim()">
              <mat-icon>send</mat-icon>
            </button>
          </mat-form-field>
        </mat-card-actions>
      </mat-card>
    </div>
  `,
  styles: [`
    /* Minimal layout CSS. Colors/Fonts are handled automatically by Angular Material */
    :host {
      display: block;
      height: 100vh;
      width: 100vw;
      overflow: hidden;
      box-sizing: border-box;
    }
    .chat-container {
      display: flex;
      justify-content: center;
      height: 100%;
      width: 100%;
      padding: 16px 20px;
      box-sizing: border-box;
    }
    .chat-card {
      width: 100%;
      max-width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      border-radius: 12px;
      overflow: hidden;
      box-sizing: border-box;
    }
    mat-card-header {
      padding: 16px 24px 12px 24px;
      border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    }
    .chat-messages {
      flex-grow: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
      padding: 20px 24px;
    }
    .message-row {
      display: flex;
      width: 100%;
    }
    .message-row.user {
      justify-content: flex-end;
    }
    .message-bubble {
      padding: 14px 18px;
      border-radius: 12px;
      max-width: 75%;
      line-height: 1.5;
    }
    .message-row.user .message-bubble {
      background-color: var(--mat-sys-primary, #3f51b5);
      color: var(--mat-sys-on-primary, white);
    }
    .message-row.assistant .message-bubble {
      background-color: #f5f5f5;
      color: rgba(0, 0, 0, 0.87);
    }
    .chat-input-area {
      padding: 12px 24px 8px 24px;
      border-top: 1px solid rgba(0, 0, 0, 0.06);
    }
    .full-width {
      width: 100%;
    }
    .source-title {
      font-size: 0.85rem;
      align-items: center;
    }
    .source-icon {
      font-size: 1.2rem;
      width: 1.2rem;
      height: 1.2rem;
      margin-right: 8px;
    }
    .source-item {
      margin-top: 12px;
      font-size: 0.85rem;
      border-left: 3px solid #ccc;
      padding-left: 12px;
    }
    .loading-indicator {
      display: flex;
      justify-content: center;
      padding: 10px;
    }

    /* Feedback styles */
    .feedback-section {
      margin-top: 8px;
      padding-top: 4px;
      border-top: 1px solid rgba(0, 0, 0, 0.08);
    }
    .feedback-actions {
      display: flex;
      align-items: center;
      gap: 4px;
    }
    .feedback-btn {
      transform: scale(0.85);
    }
    .feedback-thanks {
      font-size: 0.75rem;
      color: rgba(0, 0, 0, 0.5);
      margin-left: 4px;
    }
    .feedback-comment-area {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 8px;
    }
    .feedback-comment-field {
      width: 100%;
      font-size: 0.85rem;
    }
    .submit-feedback-btn {
      align-self: flex-end;
      min-width: 140px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .btn-spinner {
      display: inline-block;
    }
  `]
})
export class RagChatComponent {
  messages: ChatMessage[] = [];
  userInput: string = '';
  isLoading: boolean = false;
  
  private apiUrl = 'http://localhost:8000/api/chat';
  private feedbackUrl = 'http://localhost:8000/api/feedback';
  private http = inject(HttpClient);
  private snackBar = inject(MatSnackBar);
  
  // 1. Inject the Change Detector
  private cdr = inject(ChangeDetectorRef); 

  sendMessage() {
    if (!this.userInput.trim()) return;

    const userText = this.userInput;
    
    // 1. Grab the last 5 messages to send as context BEFORE adding the new user message
    const historyToSend = this.messages.slice(-5).map(msg => ({
      role: msg.role,
      content: msg.content
    }));

    this.messages = [...this.messages, { role: 'user', content: userText }];
    this.userInput = '';
    this.isLoading = true;

    // 2. Include the history array in the HTTP request payload
    this.http.post<any>(this.apiUrl, { 
      question: userText, 
      history: historyToSend 
    }).subscribe({
      next: (response) => {
        this.messages = [
          ...this.messages, 
          {
            role: 'assistant',
            content: response.answer,
            sources: response.sources,
            question: userText,
            feedback: null,
            feedbackSubmitting: false
          }
        ];
        
        this.isLoading = false;
        this.cdr.detectChanges(); 
        this.scrollToBottom();
      },
      error: (err) => {
        this.messages = [...this.messages, { role: 'assistant', content: 'Error: Could not connect.' }];
        this.isLoading = false;
        this.cdr.detectChanges(); 
      }
    });
  }

  trackByIndex(index: number): number {
    return index;
  }

  toggleRating(messageIndex: number, rating: 'thumbs_up' | 'thumbs_down') {
    const msg = this.messages[messageIndex];
    if (!msg || msg.role !== 'assistant') return;

    // Toggle off if same rating clicked again
    if (msg.feedback === rating) {
      this.messages[messageIndex] = { ...msg, feedback: null, feedbackComment: '', showCommentBox: false };
    } else {
      // Set the rating and show comment box
      this.messages[messageIndex] = { ...msg, feedback: rating, showCommentBox: true };
    }
    this.cdr.detectChanges();
  }

  updateComment(messageIndex: number, event: Event) {
    const value = (event.target as HTMLTextAreaElement).value;
    this.messages[messageIndex].feedbackComment = value;
  }

  submitFeedback(messageIndex: number) {
    const msg = this.messages[messageIndex];
    if (!msg || msg.role !== 'assistant' || !msg.feedback) return;

    // Mark as submitting
    this.messages[messageIndex] = { ...msg, feedbackSubmitting: true };
    this.cdr.detectChanges();

    const payload = {
      rating: msg.feedback,
      comment: msg.feedbackComment || null,
      question: msg.question || null,
      answer: msg.content
    };

    this.http.post<any>(this.feedbackUrl, payload).subscribe({
      next: () => {
        this.messages[messageIndex] = {
          ...this.messages[messageIndex],
          feedbackSubmitting: false,
          feedbackSent: true,
          showCommentBox: false
        };
        this.cdr.detectChanges();
      },
      error: () => {
        this.messages[messageIndex] = {
          ...this.messages[messageIndex],
          feedbackSubmitting: false
        };
        this.snackBar.open('Failed to submit feedback', 'Dismiss', { duration: 3000 });
        this.cdr.detectChanges();
      }
    });
  }

  private scrollToBottom() {
    setTimeout(() => {
      const chatContainer = document.querySelector('.chat-messages');
      if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }
    }, 100);
  }
}
