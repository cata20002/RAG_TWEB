import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { RagChatComponent } from '../rag-chat/rag-chat';

@Component({
  imports: [RouterOutlet, RagChatComponent],
  selector: 'app-root',
  standalone: true,
  styleUrl: './app.scss',
  templateUrl: './app.html',
})
export class App {
  protected readonly title = signal('rag-fe');
}
